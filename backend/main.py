from pathlib import Path
from io import BytesIO
import io, json
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; MODELS=ROOT/'models'
MONTHS=['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
FEATURES=['municipio','mes_num','mes_sin','mes_cos','anio','dias_fiesta','dias_ballenas','dias_festivo_nac','puentes','semana_santa_dias']
app=FastAPI(title='CHOCÓDATA · Análisis integrado y predicción',version='2.0')
app.mount('/static',StaticFiles(directory=ROOT/'frontend'),name='static')

@app.get('/')
def home():return FileResponse(ROOT/'frontend'/'index.html')

def readcsv(filename):return pd.read_csv(DATA/filename)

def records(df):return json.loads(df.replace({np.nan:None}).to_json(orient='records',force_ascii=False))

@app.get('/api/resumen')
def resumen():
    d=readcsv('integrado.csv'); f=readcsv('pronosticos.csv'); e=readcsv('evaluacion.csv')
    annual=d[d.anio<=2025].groupby('anio',as_index=False).llegadas.sum()
    municipal=d[d.anio==2025].groupby('municipio',as_index=False).llegadas.sum().sort_values('llegadas',ascending=False)
    source_coverage={'registros_integrados':len(d),'coincidencias_extranjeros':int(d.extranjeros_no_residentes_registrados.notna().sum()),'municipios':int(d.municipio.nunique())}
    perf=e[e.division.isin(['Prueba 2025','Seguimiento 2026 (ene-jun)'])][['division','modelo','WAPE_pct','MAE','acierto_demanda_pct']]
    return {'cobertura':source_coverage,'anual':records(annual),'ranking':records(municipal),'evaluacion':records(perf),'pronosticos':len(f)}

@app.get('/api/historico')
def historico():
    d=readcsv('integrado.csv'); agg=d.groupby(['anio','mes_num'],as_index=False).llegadas.sum()
    return {'total':len(d),'municipios':sorted(d.municipio.unique().tolist()),'serie':records(agg),'cobertura':records(d.groupby('municipio',as_index=False).size().rename(columns={'size':'meses'}))}

@app.get('/api/pronosticos')
def pronosticos(municipio:str|None=None,anio:int|None=None):
    d=readcsv('pronosticos.csv')
    if municipio:d=d[d.municipio==municipio.upper().strip()]
    if anio:d=d[d.anio==anio]
    out=d[['municipio','anio','mes_num','estimacion_random_forest','demanda_random_forest','estimacion_estacional','demanda_estacional','coinciden_categorias']].copy()
    out.insert(3,'mes',out.mes_num.map(lambda m:MONTHS[m-1]))
    return {'resultados':records(out),'total':len(out),'alcance':'Pronóstico de llegadas aéreas, indicador de actividad turística'}

@app.get('/api/modelo')
def modelo():
    x=json.loads((MODELS/'metadatos.json').read_text(encoding='utf-8'))
    return {'version':'2.0','algoritmo':'Random Forest + referencia estacional','target':'Llegadas aéreas mensuales','variables':x['feature_columns'],'entrenado_hasta':x['entrenado_hasta'],'registros_entrenamiento':x['n_registros_entrenamiento'],'municipios_aptos':sorted(readcsv('integrado.csv').municipio.unique().tolist()),'evaluacion':(pd.read_csv(DATA/'evaluacion.csv').query("division == 'Prueba 2025'")[['modelo','WAPE_pct','acierto_demanda_pct']]).to_dict('records')}

async def load(file):
    b=await file.read()
    if not b or len(b)>10_000_000:raise HTTPException(400,'Archivo vacío o superior a 10 MB')
    try:
        if (file.filename or '').lower().endswith('.csv'):return pd.read_csv(BytesIO(b),sep=None,engine='python')
        if (file.filename or '').lower().endswith('.xlsx'):return pd.read_excel(BytesIO(b))
    except Exception:raise HTTPException(400,'No se pudo leer el archivo; revisa el formato.')
    raise HTTPException(400,'Utiliza un archivo CSV o XLSX.')

def check(df):
    required=['municipio','anio','mes_num'];errors=[];warnings=[]
    for col in required:
        if col not in df:errors.append(f'Falta la columna {col}')
    if errors:return errors,warnings
    cities=set(readcsv('integrado.csv').municipio.unique())
    if df[required].isna().any().any():errors.append('No se permiten valores vacíos en municipio, año o mes')
    for col in ['anio','mes_num']:
        n=pd.to_numeric(df[col],errors='coerce');bad=n.isna()|(~np.isfinite(n))|(n%1!=0)
        if bad.any():errors.append(f'{col} debe ser entero ({int(bad.sum())} registros)')
        elif col=='mes_num' and (~n.between(1,12)).any():errors.append('mes_num debe estar entre 1 y 12')
        elif col=='anio' and (~n.between(2026,2027)).any():errors.append('La demostración de predicción admite 2026–2027')
    unknown=set(df.municipio.astype(str).str.strip().str.upper())-cities
    if unknown:errors.append('Municipios fuera de cobertura: '+', '.join(sorted(unknown)))
    if df[required].duplicated().any():errors.append('Hay municipio-año-mes duplicados')
    if any(c not in required for c in df.columns):warnings.append('Las columnas adicionales no se utilizan para predecir')
    if not errors:
        allowed={(int(a),int(m)) for a,m in readcsv('festividades.csv')[['anio','mes_num']].drop_duplicates().itertuples(index=False,name=None)}
        outside=[(int(r.anio),int(r.mes_num)) for r in df.itertuples() if (int(r.anio),int(r.mes_num)) not in allowed or (int(r.anio)==2026 and int(r.mes_num)<=6)]
        if outside:errors.append('Solo se admiten meses futuros con calendario documentado: julio 2026 a diciembre 2027')
    return errors,warnings

@app.post('/api/validar')
async def validar(file:UploadFile=File(...)):
    df=await load(file);er,wa=check(df)
    return {'estado':'Incompatible' if er else 'Compatible con advertencias' if wa else 'Compatible','compatible':not bool(er),'filas':len(df),'errores':er,'advertencias':wa,'columnas':list(df.columns)}

@app.post('/api/eda')
async def eda(file:UploadFile=File(...)):
    df=await load(file)
    n=df.select_dtypes(include='number')
    return {'filas':len(df),'columnas':len(df.columns),'duplicados':int(df.duplicated().sum()),'faltantes':records(df.isna().sum().rename('faltantes').rename_axis('variable').reset_index()),'numericas':records(n.describe().round(2).T.reset_index(names='variable')) if len(n.columns) else [],'categoricas':{c:records(df[c].fillna('(sin dato)').astype(str).value_counts().head(10).rename_axis('valor').reset_index(name='cantidad')) for c in df.select_dtypes(exclude='number').columns[:8]}}

@app.post('/api/predecir')
async def predecir(file:UploadFile=File(...)):
    df=await load(file);errors,warnings=check(df)
    if errors:raise HTTPException(422,detail={'mensaje':'Archivo incompatible. No se generaron predicciones.','errores':errors})
    df=df[['municipio','anio','mes_num']].copy();df.municipio=df.municipio.str.strip().str.upper();df.anio=df.anio.astype(int);df.mes_num=df.mes_num.astype(int)
    events=readcsv('festividades.csv');df=df.merge(events[['municipio','anio','mes_num','dias_fiesta','dias_ballenas','dias_festivo_nac','puentes','semana_santa_dias']],on=['municipio','anio','mes_num'],how='left',validate='many_to_one')
    if df[FEATURES[5:]].isna().any().any():raise HTTPException(422,'Falta calendario para el período solicitado')
    df['mes_sin']=np.sin(2*np.pi*df.mes_num/12);df['mes_cos']=np.cos(2*np.pi*df.mes_num/12)
    model=joblib.load(MODELS/'modelo_rf.joblib');values=np.maximum(0,np.expm1(model.predict(df[FEATURES])))
    reference=readcsv('pronosticos.csv')[['municipio','anio','mes_num','demanda_random_forest','estimacion_estacional','demanda_estacional']]
    out=df[['municipio','anio','mes_num']].copy();out['estimacion_random_forest']=np.rint(values).astype(int)
    out=out.merge(reference,on=['municipio','anio','mes_num'],how='left',validate='many_to_one')
    if out.demanda_random_forest.isna().any():raise HTTPException(422,'Faltan umbrales de demanda documentados')
    return {'registros':len(out),'resultados':records(out),'nota':'Predicción de llegadas aéreas como indicador de actividad turística','advertencias':warnings}

@app.get('/api/descargar')
def descargar(tipo:str='pronosticos'):
    mapping={'pronosticos':'pronosticos.csv','historico':'historico.csv','integracion':'integrado.csv','evaluacion':'evaluacion.csv'}
    if tipo not in mapping:raise HTTPException(404,'Reporte no disponible')
    return FileResponse(DATA/mapping[tipo],filename='chocodata_'+tipo+'.csv')

@app.get('/api/salud')
def salud():return {'estado':'ok','version':'2.0','modo':'demostracion publica, solo lectura de modelos'}
