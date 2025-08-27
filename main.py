import os, re, io, json, uuid, requests, contextvars
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import msal
from fastapi.responses import JSONResponse
import traceback
from fastapi import HTTPException
from dotenv import load_dotenv
from urllib.parse import quote
from google import genai
from google.genai import types as gtypes

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

GRAPH_BASE = os.getenv("GRAPH_BASE", "https://graph.microsoft.com/v1.0")
TENANT_ID = os.getenv("TENANT_ID")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if o.strip()
]

DATA_INSIGHT_SCHEMA = gtypes.Schema(
    type=gtypes.Type.OBJECT,
    properties={
        "answer": gtypes.Schema(type=gtypes.Type.STRING),
        "anomalies": gtypes.Schema(
            type=gtypes.Type.ARRAY,
            items=gtypes.Schema(
                type=gtypes.Type.OBJECT,
                properties={
                    "row_index": gtypes.Schema(type=gtypes.Type.INTEGER),
                    "score": gtypes.Schema(type=gtypes.Type.NUMBER),
                    "values": gtypes.Schema(
                        type=gtypes.Type.ARRAY,
                        items=gtypes.Schema(
                            type=gtypes.Type.OBJECT,
                            properties={
                                "column": gtypes.Schema(type=gtypes.Type.STRING),
                                "value": gtypes.Schema(type=gtypes.Type.NUMBER),
                            },
                            required=["column", "value"],
                        ),
                    ),
                },
                required=["row_index"],
            ),
        ),
        "recommendations": gtypes.Schema(
            type=gtypes.Type.ARRAY,
            items=gtypes.Schema(type=gtypes.Type.STRING),
        ),
        "citations": gtypes.Schema(
            type=gtypes.Type.ARRAY,
            items=gtypes.Schema(type=gtypes.Type.STRING),
        ),
    },
    required=["answer"],
)

app = FastAPI(title="Data Agent API · Gemini + SharePoint")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)
SESSIONS: Dict[str, Dict[str, Any]] = {}
CURRENT_SESSION_ID = contextvars.ContextVar("CURRENT_SESSION_ID", default=None)


@app.middleware("http")
async def cors_on_all_responses(request, call_next):
    origin = request.headers.get("origin")
    try:
        response = await call_next(request)
    except Exception as e:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print("ERROR / Exception:\n", tb)
        response = JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": str(e)},
        )
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.options("/{rest_of_path:path}")
def preflight(rest_of_path: str, request: Request):
    origin = request.headers.get("origin", "")
    headers = {
        "Access-Control-Allow-Origin": origin if origin in ALLOWED_ORIGINS else "",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": request.headers.get(
            "access-control-request-headers", "*"
        ),
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
    }
    return Response(status_code=204, headers=headers)


def profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    prof = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [],
        "na_rate": df.isna().mean(numeric_only=False).round(4).to_dict(),
    }
    for col in df.columns:
        s = df[col]
        info = {
            "name": str(col),
            "dtype": str(s.dtype),
            "sample_values": s.dropna().astype(str).head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            for k in ["min", "max", "mean", "std"]:
                v = desc.get(k, np.nan)
                info[k] = None if pd.isna(v) else float(v)
        prof["columns"].append(info)
    return prof


def value_counts(df: pd.DataFrame, column: str, top_n: int = 10):
    vc = df[column].astype(str).value_counts().head(top_n)
    return [{"value": k, "count": int(v)} for k, v in vc.items()]


def corr_matrix(df: pd.DataFrame, cols: Optional[list[str]] = None):
    num = (df[cols] if cols else df).select_dtypes(include=[np.number])
    corr = (
        num.corr(numeric_only=True).fillna(0).round(4)
        if not num.empty
        else pd.DataFrame()
    )
    return corr.to_dict()


def detect_anom(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    contamination: float = 0.02,
    limit: int = 200,
):
    num = (df[cols] if cols else df).select_dtypes(include=[np.number]).dropna()
    if num.empty or num.shape[0] < 10:
        return {"note": "Insuficientes datos numéricos para anomalías", "anomalies": []}
    iso = IsolationForest(
        n_estimators=200, contamination=float(contamination), random_state=7
    )
    labels = iso.fit_predict(num.values)
    scores = iso.decision_function(num.values)
    idx = np.where(labels == -1)[0][:limit]
    anomalies = []
    for i in idx:
        row = num.iloc[i]
        anomalies.append(
            {
                "row_index": int(row.name),
                "score": float(scores[i]),
                "values": {
                    c: (None if pd.isna(v) else float(v))
                    for c, v in row.to_dict().items()
                },
            }
        )
    return {
        "count": len(anomalies),
        "columns_used": list(num.columns),
        "anomalies": anomalies,
    }


def filter_and_describe(
    sheet: str,
    filters: List[Dict[str, Any]],
    return_rows: bool = False,
    limit: int = 20,
) -> Dict[str, Any]:
    try:
        df = _get_df(sheet).copy()
        original_rows = df.shape[0]

        for f in filters:
            col, op, val = f.get("column"), f.get("operator"), f.get("value")

            if col not in df.columns:
                return {"error": f"La columna '{col}' no existe en la hoja '{sheet}'."}

            # ROBUSTEZ: Convertir la columna y el valor a string para la comparación, evitando errores de tipo.
            # Esto es especialmente útil para operadores de igualdad como 'eq', 'neq', 'in', 'notin'.
            # Para operadores numéricos, asumimos que los tipos son correctos, pero pandas suele manejarlo.
            if op == "eq":
                # Usamos .str.contains() para una búsqueda flexible, insensible a mayúsculas/minúsculas
                if pd.api.types.is_string_dtype(df[col]):
                    condition = df[col].str.contains(str(val), case=False, na=False)
                else:  # Mantenemos la igualdad estricta para números o fechas
                    condition = df[col] == val
                df = df[condition]
            # ... (el resto de los operadores 'neq', 'gt', etc. pueden quedar igual) ...

            if op == "neq":
                # Compara texto con texto para evitar problemas de int vs str (30 vs "30")
                condition = df[col].astype(str) == str(val)
                if op == "neq":
                    condition = ~condition
                df = df[condition]
            elif op in ["in", "notin"]:
                # Asegurarse de que los valores en la lista de comparación también sean strings
                compare_values = [str(v) for v in val]
                condition = df[col].astype(str).isin(compare_values)
                if op == "notin":
                    condition = ~condition
                df = df[condition]
            # Para comparaciones numéricas, el casteo directo puede dar error si hay texto.
            # Lo dejamos como estaba, ya que una excepción aquí será capturada abajo.
            elif op == "gt":
                df = df[df[col] > val]
            elif op == "lt":
                df = df[df[col] < val]
            elif op == "gte":
                df = df[df[col] >= val]
            elif op == "lte":
                df = df[df[col] <= val]
            else:
                return {"error": f"Operador no soportado: '{op}'"}

            # print(f"DEBUG: Después de filtrar por '{col} {op} {val}', quedan {df.shape[0]} filas.")

        # MANEJO DE CASO VACÍO: Si el df queda vacío, no intentes perfilarlo.
        if df.empty:
            return {
                "note": "No se encontraron filas que coincidan con los filtros aplicados.",
                "filters_applied": filters,
                "original_rows": original_rows,
                "filtered_rows": 0,
            }

        profile = profile_df(df)
        profile["filtered_rows"] = df.shape[0]
        profile["original_rows"] = original_rows

        result = {"profile": profile}
        if return_rows:
            result["rows"] = df.head(limit).to_dict(orient="records")

        return result

    except Exception as e:
        # CAPTURA DE ERROR: Si algo falla, lo imprimimos en consola y le decimos al modelo qué pasó.
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(
            f"--- ERROR INTERNO EN filter_and_describe ---\n{tb_str}\n-------------------------------------------"
        )
        return {
            "error": f"Se produjo una excepción inesperada en la herramienta: {str(e)}"
        }


# ===== NUEVA HERRAMIENTA DE ALTO NIVEL =====


def get_rows_above_group_average_tool(
    sheet: str,
    group_column: str,
    value_column: str,
    label_column: str,
) -> Dict[str, Any]:
    """
    Encuentra el grupo más reciente/grande, calcula el promedio de una columna de valor para ese grupo,
    y devuelve los ítems cuya valor en esa columna supera dicho promedio.
    Ideal para preguntas como 'qué [label_column] estuvieron sobre el promedio de [value_column] en la última [group_column]'.

    Args:
        sheet: Nombre de la hoja a analizar.
        group_column: La columna para agrupar y encontrar el valor máximo (ej. 'week').
        value_column: La columna numérica sobre la que se calculará el promedio (ej. 'hours', 'cpu_usage').
        label_column: La columna que contiene las etiquetas o nombres a devolver (ej. 'jobname').
    """
    try:
        df = _get_df(sheet)

        # 1. Encontrar el valor máximo del grupo (ej. la última semana)
        if not pd.api.types.is_numeric_dtype(df[group_column]):
            return {"error": f"La columna de grupo '{group_column}' debe ser numérica."}
        max_group_value = df[group_column].max()

        # 2. Filtrar el DataFrame para ese grupo
        last_week_df = df[df[group_column] == max_group_value].copy()
        if last_week_df.empty:
            return {
                "note": f"No se encontraron datos para el grupo más reciente ({group_column} = {max_group_value})."
            }

        # 3. Calcular el promedio de la columna de valor
        if not pd.api.types.is_numeric_dtype(last_week_df[value_column]):
            return {
                "error": f"La columna de valor '{value_column}' debe ser numérica para calcular el promedio."
            }
        average_value = last_week_df[value_column].mean()

        # 4. Filtrar para encontrar las filas por encima del promedio
        above_average_df = last_week_df[last_week_df[value_column] > average_value]

        if above_average_df.empty:
            return {
                "note": "No se encontraron ítems por encima del promedio.",
                "group_value": float(max_group_value),
                "average_calculated": float(average_value),
                "items_above_average": [],
            }

        # 5. Extraer las etiquetas
        items_above_average = above_average_df[label_column].unique().tolist()

        return {
            "group_value_found": float(max_group_value),
            "average_calculated": float(average_value),
            "items_above_average": items_above_average,
            "count": len(items_above_average),
        }

    except KeyError as e:
        return {"error": f"No se encontró la columna: {str(e)}."}
    except Exception as e:
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"--- ERROR INTERNO EN get_rows_above_group_average_tool ---\n{tb_str}")
        return {"error": f"Se produjo una excepción inesperada: {str(e)}"}


# ===== TOOLS (Function Calling) =====
def _get_df(sheet: str) -> pd.DataFrame:
    sid = CURRENT_SESSION_ID.get()
    print(f"DEBUG (_get_df): Buscando sesión con ID: {sid}")  # <--- AÑADIR
    if not sid or sid not in SESSIONS:
        print(f"DEBUG (_get_df): ¡Sesión '{sid}' NO encontrada!")  # <--- AÑADIR
        raise RuntimeError("Sesión no encontrada")

    dfs: Dict[str, pd.DataFrame] = SESSIONS[sid]["dfs"]
    print(
        f"DEBUG (_get_df): Hojas disponibles en la sesión: {list(dfs.keys())}"
    )  # <--- AÑADIR

    if sheet not in dfs:
        print(
            f"DEBUG (_get_df): ¡Hoja '{sheet}' NO encontrada en la sesión!"
        )  # <--- AÑADIR
        raise RuntimeError(f"Hoja '{sheet}' no existe")

    df = dfs[sheet]
    print(
        f"DEBUG (_get_df): DataFrame '{sheet}' encontrado con {df.shape[0]} filas."
    )  # <--- AÑADIR
    return df


def list_sheets() -> Dict[str, Any]:
    """Lista las hojas disponibles en la sesión actual."""
    sid = CURRENT_SESSION_ID.get()
    if not sid or sid not in SESSIONS:
        return {"sheets": []}
    return {"sheets": list(SESSIONS[sid]["dfs"].keys())}


def profile_sheet(sheet: str) -> Dict[str, Any]:
    """Perfilado de una hoja (schema, NA, stats básicas)."""
    df = _get_df(sheet)
    profile_data = profile_df(df)
    print(
        f"DEBUG (profile_sheet): Devolviendo perfil para '{sheet}': {json.dumps(profile_data, indent=2)}"
    )  # <--- AÑADIR
    return profile_data


def get_graph_access_token() -> str:
    if not all([TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET]):
        raise ValueError(
            "Las variables de entorno de Microsoft (TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET) no estan correctamente configuradas"
        )

    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    app = msal.ConfidentialClientApplication(
        MS_CLIENT_ID, authority=authority, client_credential=MS_CLIENT_SECRET
    )

    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    if not result or "access_token" not in result:
        raise RuntimeError(
            f"No se pudo obtener el token de acceso: "
            f"{result.get('error')} | {result.get('error_description')}"
        )

    return result["access_token"]


def get_sharepoint_file_content(site_name: str, file_path: str) -> bytes:
    """
    Encuentra un sitio de SharePoint por su nombre y descarga un archivo por su ruta.

    Args:
        site_name: El nombre del sitio en la URL (ej. 'ITSITE').
        file_path: La ruta relativa del archivo.
    """
    token = get_graph_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    hostname = os.getenv("SHAREPOINT_HOSTNAME")
    if not hostname:
        raise ValueError(
            "La variable de entorno SHAREPOINT_HOSTNAME no está configurada en el archivo .env"
        )

    # CAMBIO CLAVE: Usamos una ruta directa en lugar de buscar. Es mucho más fiable.
    # Esto construye una ruta como: /sites/totalcivilconstruction1.sharepoint.com:/sites/ITSITE
    site_url_segment = f"{hostname}:/sites/{site_name}"

    # 1. Obtener el ID del sitio directamente
    get_site_url = f"{GRAPH_BASE}/sites/{site_url_segment}"
    response = requests.get(get_site_url, headers=headers)
    response.raise_for_status()  # Lanzará un error si el sitio no se encuentra
    site_id = response.json()["id"]

    # 2. Descargar el contenido del archivo (esta parte ya estaba bien)
    encoded_file_path = quote(file_path)
    download_url = (
        f"{GRAPH_BASE}/sites/{site_id}/drive/root:/{encoded_file_path}:/content"
    )

    print(f"DEBUG: URL de descarga construida: {download_url}")

    response = requests.get(download_url, headers=headers)

    # Añadimos más detalles al error si falla la descarga
    if response.status_code != 200:
        print(
            f"ERROR: Falló la descarga del archivo. Status: {response.status_code}, Body: {response.text}"
        )

    response.raise_for_status()

    return response.content


# Reemplaza tu función load_file_from_sharepoint_tool con esta versión mejorada


def load_file_from_sharepoint_tool(site_name: str, file_path: str) -> Dict[str, Any]:
    """
    Carga un archivo Excel desde un sitio de SharePoint en una nueva sesión de análisis.
    Usa esta herramienta cuando el usuario pida analizar un archivo ubicado en SharePoint.
    """
    try:
        print(
            f"DEBUG: Intentando cargar el archivo '{file_path}' desde el sitio '{site_name}' de SharePoint."
        )

        sid = CURRENT_SESSION_ID.get()
        if not sid:
            raise RuntimeError(
                "No se pudo obtener el ID de la sesión activa desde el contexto."
            )

        print(
            f"DEBUG: Intentando cargar el archivo '{file_path}' en la sesión existente '{sid}'."
        )
        file_content = get_sharepoint_file_content(site_name, file_path)

        sid = CURRENT_SESSION_ID.get()
        xls = pd.ExcelFile(io.BytesIO(file_content))
        dfs = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}

        SESSIONS[sid] = {
            "dfs": dfs,
            "meta": {"filename": file_path, "source": "SharePoint", "site": site_name},
        }

        print(
            f"DEBUG: Archivo cargado exitosamente en la sesion {sid}. Hojas: {list(dfs.keys())}"
        )

        CURRENT_SESSION_ID.set(sid)

        return {
            "status": "success",
            "session_id": sid,
            "sheets_found": list(dfs.keys()),
            "message": f"El archivo '{file_path}' se cargo correctamente. Ahora puede usar la herramienta de analisis en esta sesion.",
        }

    # CAMBIO CLAVE: Manejo de errores más específico y detallado
    except requests.exceptions.HTTPError as http_err:
        error_details = http_err.response.text
        print(f"--- ERROR HTTP DETALLADO DESDE MICROSOFT GRAPH ---")
        print(error_details)
        print(f"-------------------------------------------------")

        # Intentamos parsear el JSON para darle un mensaje más limpio al modelo
        try:
            error_json = http_err.response.json()
            error_message = error_json.get("error", {}).get(
                "message", "Error HTTP desconocido."
            )
        except json.JSONDecodeError:
            error_message = error_details

        return {
            "status": "error",
            "message": f"Error de Microsoft Graph: {error_message}",
        }
    except Exception as e:
        # Captura para otros errores (ej. archivo no encontrado, problemas de token)
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print(f"--- ERROR INTERNO en load_file_from_sharepoint_tool ---\n{tb_str}")
        return {
            "status": "error",
            "message": f"No se pudo cargar el archivo: {str(e)}",
        }


def value_counts_tool(sheet: str, column: str, top_n: int = 10) -> Dict[str, Any]:
    """Top-N conteos de una columna.
    Args:
      sheet: nombre de la hoja
      column: nombre de la columna
      top_n: cantidad máxima (default 10)
    """
    return {"rows": value_counts(_get_df(sheet), column, top_n)}


def corr_tool(sheet: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Matriz de correlación numérica (Pearson).
    Args:
      sheet: nombre de la hoja
      columns: lista opcional de columnas a considerar
    """
    return {"corr": corr_matrix(_get_df(sheet), columns)}


def detect_anomalies_tool(
    sheet: str, columns: Optional[List[str]] = None, contamination: float = 0.02
) -> Dict[str, Any]:
    """Detecta anomalías con IsolationForest.
    Args:
      sheet: nombre de la hoja
      columns: columnas numéricas a usar (opcional)
      contamination: proporción esperada de outliers (0.0–0.5)
    """
    return detect_anom(_get_df(sheet), columns, contamination)


# ===== NUEVA HERRAMIENTA DE ANÁLISIS AGRUPADO =====


def analyze_group_performance_tool(
    sheet: str,
    group_column: str,
    group_value: str,
    value_column: str,
    label_column: str,
) -> Dict[str, Any]:
    """
    Filtra los datos por un valor específico en una columna de grupo, calcula el promedio de una columna de valor
    para ESE GRUPO, y devuelve las etiquetas de las filas que superan ese promedio.
    Es ideal para preguntas como 'qué [empleados] superaron el promedio de [horas] para el [job] específico'.

    Args:
        sheet: Nombre de la hoja a analizar.
        group_column: La columna por la que se agrupará (ej. 'Job Name').
        group_value: El valor específico del grupo a analizar (ej. 'Spotsylvania').
        value_column: La columna numérica sobre la que se calculará el promedio (ej. 'HOURS').
        label_column: La columna que contiene las etiquetas o nombres a devolver (ej. 'Employees Name').
    """
    try:
        df = _get_df(sheet)

        # 1. Filtrar el grupo de forma flexible (case-insensitive, partial match)
        group_df = df[
            df[group_column].str.contains(group_value, case=False, na=False)
        ].copy()
        if group_df.empty:
            return {
                "note": f"No se encontraron datos para el grupo '{group_value}' en la columna '{group_column}'."
            }

        # 2. Calcular el promedio de ese grupo específico
        if not pd.api.types.is_numeric_dtype(group_df[value_column]):
            return {"error": f"La columna de valor '{value_column}' debe ser numérica."}
        group_average = group_df[value_column].mean()

        # 3. Encontrar las filas del grupo que superan el promedio del grupo
        performers = group_df[group_df[value_column] > group_average]
        if performers.empty:
            return {
                "note": "Dentro del grupo especificado, ningún ítem superó el promedio.",
                "group_analyzed": group_value,
                "group_average": float(group_average),
            }

        # 4. Devolver las etiquetas de los que superaron el promedio
        performer_labels = performers[label_column].unique().tolist()
        return {
            "group_analyzed": group_value,
            "group_average": float(group_average),
            "items_above_average": performer_labels,
            "count": len(performer_labels),
        }
    except KeyError as e:
        return {"error": f"No se encontró la columna: {str(e)}."}
    except Exception as e:
        return {"error": f"Se produjo una excepción inesperada: {str(e)}"}


filter_and_describe_tool = filter_and_describe

TOOLS = [
    load_file_from_sharepoint_tool,
    list_sheets,
    profile_sheet,
    value_counts_tool,
    corr_tool,
    analyze_group_performance_tool,
    detect_anomalies_tool,
    filter_and_describe_tool,
    get_rows_above_group_average_tool,
]  # auto tool-calling


# ===== Salida estructurada =====
class DataInsight(BaseModel):
    answer: str = Field(..., description="Respuesta en lenguaje natural al usuario")
    anomalies: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Lista de anomalías (si aplica)"
    )
    recommendations: Optional[List[str]] = Field(
        default=None, description="Acciones recomendadas, priorizadas (3–7)"
    )
    citations: Optional[List[str]] = None


# ===== Endpoints =====
@app.post("/upload")
async def upload_excel(
    file: UploadFile = File(...), session_id: Optional[str] = Form(None)
):
    sid = session_id or str(uuid.uuid4())
    data = await file.read()
    xls = pd.ExcelFile(io.BytesIO(data))
    dfs = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
    SESSIONS[sid] = {"dfs": dfs, "meta": {"filename": file.filename}, "history": []}
    return {"session_id": sid, "sheets": list(dfs.keys())}


class AskBody(BaseModel):
    session_id: Optional[str] = None
    question: str
    sheet: Optional[str] = None


@app.post("/ask")
def ask(body: AskBody):
    sid = body.session_id

    if not sid or sid not in SESSIONS:
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {"dfs": {}, "meta": {}, "history": []}

    CURRENT_SESSION_ID.set(sid)

    system_prompt = (
        "Eres un analista de datos senior, eficiente y orientado a la acción. Tu objetivo principal es responder las preguntas del usuario con datos concretos, no con conversaciones vacías.\n"
        "Sigue estas reglas de comportamiento:\n"
        "1. **Verifica el contexto actual**: Antes de actuar, usa `list_sheets` para saber si ya hay un archivo cargado en la sesión.\n"
        "2. **Si ya hay un archivo**: Usa las herramientas de análisis (`profile_sheet`, `filter_and_describe`, etc.) para responder directamente. NO intentes volver a cargar un archivo que ya está en la sesión.\n"
        "3. **Si no hay archivo y el usuario lo pide**: Usa `load_file_from_sharepoint_tool` para cargarlo.\n"
        "4. **USA LA HERRAMIENTA CORRECTA**: Para preguntas simples de filtrado, usa `filter_and_describe`. Para preguntas complejas que piden comparar ítems contra el promedio de SU PROPIO GRUPO (ej. 'empleados sobre el promedio de su job' o 'productos sobre el promedio de su categoría'), usa la herramienta especializada `analyze_group_performance_tool`.\n"
        "5. **LÓGICA DE CORRECCIÓN AUTOMÁTICA DE ERRORES (MUY IMPORTANTE)**:\n"
        "   a. Si una herramienta falla con un error que sugiere un nombre de columna incorrecto (como un KeyError), no te detengas. Tu siguiente paso DEBE ser llamar a `profile_sheet` para obtener la lista de los nombres de columna exactos.\n"
        "   b. Compara el nombre de columna incorrecto que se usó con la lista de columnas correctas que obtuviste. Identifica la columna correcta que más se parezca (ej. 'job name' se parece a 'Job Name', 'week' se parece a 'Week').\n"
        "   c. Una vez identificada la columna correcta, **vuelve a intentar automáticamente la herramienta original que falló**, pero esta vez con el nombre de columna corregido.\n"
        "   d. En tu respuesta final al usuario, DEBES informarle de la corrección que realizaste. Por ejemplo: \"Detecté que 'job name' estaba mal escrito y lo corregí a 'Job Name' para poder ejecutar la consulta. El resultado es el siguiente: [...]\".\n"
        "   e. Si no encuentras una columna que se parezca lo suficiente, entonces sí debes preguntar al usuario para que aclare.\n"
        "6. **Sé proactivo**: Después de cargar un archivo o realizar un análisis, sugiere el siguiente paso lógico (ej. 'El archivo se ha cargado. ¿Quieres que realice un perfilado de los datos?')."
    )

    history = SESSIONS[sid].get("history", [])
    user_text = body.question + (
        f" (hoja sugerida: {body.sheet})" if body.sheet else ""
    )
    history.append(gtypes.Content(role="user", parts=[gtypes.Part(text=user_text)]))

    # --- PASO 1: Análisis con Herramientas ---
    # En esta llamada, permitimos que el modelo use las herramientas libremente.
    # Quitamos response_mime_type y response_schema.
    try:
        analysis_resp = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=gtypes.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=TOOLS,  # Permitimos el uso de herramientas
                temperature=0.2,
                automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=4
                ),
            ),
        )
        print("DEBUG (ask): Respuesta completa del modelo (Paso 1):")
        print(analysis_resp)
        # El resultado aquí es el análisis completo en texto, después de usar las herramientas.
        analysis_text = analysis_resp.text
        history.append(analysis_resp.candidates[0].content)
        SESSIONS[sid]["history"] = history
        print(f"DEBUG: Texto de análisis generado para formatear: '{analysis_text}'")

    except Exception as e:
        print("Gemini error (Paso 1 - Análisis):", e)
        raise HTTPException(status_code=502, detail=f"Gemini analysis error: {e}")

    # --- PASO 2: Formateo a JSON ---
    # Ahora, tomamos el texto del análisis y le pedimos al modelo que lo estructure.
    # En esta llamada usamos el modo JSON, pero SIN herramientas.
    formatting_prompt = (
        "A partir del siguiente análisis de datos, extrae la información y formatea la salida estrictamente como un objeto JSON "
        "que cumpla con el esquema proporcionado. La respuesta debe ser únicamente el JSON, sin texto adicional.\n\n"
        "IMPORTANTE: Si el análisis indica que no se encontraron datos o filas, formula una respuesta amigable para el usuario en el campo 'answer', "
        "explicando qué se buscó y que no se encontraron resultados. Por ejemplo: 'Busqué los trabajos con más de 100 horas pero no encontré ninguno que cumpliera ese criterio.'\n\n"
        f"ANÁLISIS A FORMATEAR:\n---\n{analysis_text}\n---"
    )

    try:
        final_resp = client.models.generate_content(
            model=MODEL,
            contents=[formatting_prompt],
            config=gtypes.GenerateContentConfig(
                # NO incluimos `tools` aquí
                response_mime_type="application/json",
                response_schema=DATA_INSIGHT_SCHEMA,
                temperature=0.0,  # Usamos 0 para que sea determinista en el formateo
            ),
        )
        final_data = json.loads(final_resp.text)
        final_data["session_id"] = sid  # Añade el ID de la sesión actual
        return final_data

    except Exception as e:
        print("Gemini error (Paso 2 - Formateo):", e)
        # Si el formateo falla, podemos devolver el análisis en texto como fallback.
        return {
            "answer": analysis_text,
            "session_id": sid,  # Añade el ID de la sesión también aquí
            "detail": "El formateo a JSON falló.",
        }
