# PROYECTO: ECONOMÍA CONDUCTUAL (REPLICADO EN PYTHON)
# MODELOS: DiD (Diferencias en Diferencias) y RDD (Regresión Discontinua)

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col

# 1. SIMULACIÓN DE DATOS PARA EXPERIMENTO BANCARIO

np.random.seed(2026)
n_clientes = 5000

# Perfil de clientes del banco
df_base = pd.DataFrame({
    'id_cliente': np.arange(n_clientes),
    'edad': np.random.randint(18, 65, n_clientes),
    'genero': np.random.choice(['M', 'F'], n_clientes),
    'ingreso_mxn': np.random.lognormal(mean=np.log(9000), sigma=0.4, size=n_clientes),
    'score_interno': np.random.randint(500, 850, n_clientes),
    'grupo': np.random.choice(['Control', 'Tratamiento'], n_clientes)
})

# Crear Panel con dummies (Periodo 0 = Antes, Periodo 1 = Después)
df_pre = df_base.copy()
df_pre['periodo'] = 0

df_post = df_base.copy()
df_post['periodo'] = 1

df_panel = pd.concat([df_pre, df_post]).reset_index(drop=True)


# 2. BEHAVIORAL ECONOMICS (EFECTOS CAUSALES)
# Efecto DiD: El tratamiento (nudge) reduce el retiro de efectivo un 15% en el periodo 1
def calcular_retiro(row):
    base = 0.85 + np.random.normal(0, 0.05)
    efecto = -0.15 if (row['grupo'] == 'Tratamiento' and row['periodo'] == 1) else 0
    return np.clip(base + efecto, 0, 1)

df_panel['pct_retiro'] = df_panel.apply(calcular_retiro, axis=1)

# Efecto RDD: Salto en consumo si score >= 700 (acceso a línea de crédito aprobada)
df_panel['consumo_mensual'] = (
    500 + 
    (df_panel['ingreso_mxn'] * 0.05) + 
    (np.where(df_panel['score_interno'] >= 700, 1500, 0)) + 
    np.random.normal(0, 300, len(df_panel))
)

# 3. PREPARACIÓN DE DUMMIES (PARA ECONOMETRÍA)

df_panel['dummy_post'] = np.where(df_panel['periodo'] == 1, 1, 0)
df_panel['dummy_tratado'] = np.where(df_panel['grupo'] == 'Tratamiento', 1, 0)
df_panel['distancia_umbral'] = df_panel['score_interno'] - 700
df_panel['dummy_mujer'] = np.where(df_panel['genero'] == 'F', 1, 0)



#MODELOS ECONOMÉTRICOS

# MODELO 1: DIFERENCIAS EN DIFERENCIAS (DiD)
# El asterisco (*) en Python crea automáticamente los efectos principales y la interacción
modelo_did = smf.ols('pct_retiro ~ dummy_tratado * dummy_post + ingreso_mxn + dummy_mujer', data=df_panel).fit()

print("\n" + "="*60)
print("   RESULTADOS DiD: EFECTO DEL NUDGE BANCARIO (CONSOLA PYTHON)")
print("="*60)
print(modelo_did.summary())

# MODELO 2: REGRESIÓN DISCONTINUA (RDD)
# Cerca del umbral (650 a 750 puntos de score)
df_rdd = df_panel[(df_panel['score_interno'] >= 650) & (df_panel['score_interno'] <= 750)].copy()
df_rdd['dummy_cruce'] = np.where(df_rdd['score_interno'] >= 700, 1, 0)

modelo_rdd = smf.ols('consumo_mensual ~ dummy_cruce + distancia_umbral + dummy_mujer', data=df_rdd).fit()

print("\n" + "="*60)
print("   RESULTADOS RDD: IMPACTO DEL CRÉDITO EN EL CONSUMO")
print("="*60)
print(modelo_rdd.summary())



# VISUALIZACIONES
# --- Gráfico A: Diferencias en Diferencias ---
# 1. Extraemos las medias exactas para cada grupo y periodo
medias = df_panel.groupby(['grupo', 'periodo'])['pct_retiro'].mean()
c0, c1 = medias[('Control', 0)], medias[('Control', 1)]
t0, t1 = medias[('Tratamiento', 0)], medias[('Tratamiento', 1)]

# 2. Calculamos el contrafactual
cf1 = t0 + (c1 - c0) 

plt.figure(figsize=(10, 6))

# 3. Dibujar las trayectorias reales
# Línea del Grupo Control
plt.plot([0, 1], [c0, c1], marker='o', markersize=8, linewidth=2.5, color='#1f77b4', label='Grupo Control (Observado)')
# Línea del Grupo Tratamiento
plt.plot([0, 1], [t0, t1], marker='o', markersize=8, linewidth=2.5, color='#d62728', label='Grupo Tratamiento (Observado)')

# 4. Dibujar la línea contrafactual (El "qué hubiera pasado")
plt.plot([0, 1], [t0, cf1], marker='s', markersize=8, linestyle='--', linewidth=2.5, color='#d62728', alpha=0.4, label='Contrafactual (Tratado sin intervención)')

# 5. Marcar el evento temporal de la intervención
plt.axvline(x=0.5, color='grey', linestyle='-.', linewidth=1.5)
plt.text(0.5, max(c0, c1, t0, t1, cf1) + 0.02, 'Momento de la Intervención', horizontalalignment='center', color='grey', fontweight='bold')

# 6. Mostrar el efecto DiD (La diferencia entre lo observado y el contrafactual en T=1)
plt.annotate('', xy=(1, t1), xytext=(1, cf1), arrowprops=dict(arrowstyle="<->", color='black', lw=2.5))
plt.text(1.03, (t1 + cf1)/2, 'Efecto Real (DiD)', color='black', fontweight='bold', va='center')

# Configuraciones de formato del gráfico
plt.title('Gráfico Teórico: Modelo de Diferencias en Diferencias (DiD)', fontsize=14, pad=15)
plt.xticks([0, 1], ['Tiempo 0\n(Antes del Experimento)', 'Tiempo 1\n(Después del Experimento)'], fontsize=11)
plt.ylabel('% de Ingreso Retirado (Media)', fontsize=11)

# Ajustar límites del eje X para dar espacio a la etiqueta del efecto
plt.xlim(-0.1, 1.25)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', framealpha=0.9)
plt.tight_layout()
plt.show()


# Gráfico RDD
plt.figure(figsize=(10, 6))
sns.regplot(data=df_panel[df_panel['score_interno'] < 700], x='score_interno', y='consumo_mensual', scatter_kws={'alpha':0.1}, color='red', label='Sin Línea de Crédito')
sns.regplot(data=df_panel[df_panel['score_interno'] >= 700], x='score_interno', y='consumo_mensual', scatter_kws={'alpha':0.1}, color='blue', label='Con Línea de Crédito')
plt.axvline(x=700, color='black', linestyle='--', label='Umbral 700 pts')
plt.title('RDD: Salto en Consumo al cruzar el Umbral de Score')
plt.xlabel('Score Crediticio Interno')
plt.ylabel('Consumo Mensual (MXN)')
plt.legend()
plt.show()


#TABLA FINAL DE REPORTE ECONOMÉTRICO
tabla_econometrica = summary_col([modelo_rdd], 
                                 stars=True, 
                                 float_format='%0.3f',
                                 model_names=['Consumo Mensual (RDD)'],
                                 info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                            'R2': lambda x: "{:.3f}".format(x.rsquared)})

print("\n" + "="*50)
print("   TABLA DE REGRESIÓN PROFESIONAL (FORMATO PAPEL)")
print("="*50)
print(tabla_econometrica)
print("Nota: *** p<0.01, ** p<0.05, * p<0.1")
