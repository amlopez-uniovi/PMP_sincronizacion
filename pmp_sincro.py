import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

mpl.use("TkAgg")

import numpy as np
from datetime import datetime, date, time
import unicodedata
import os
import json
from scipy import signal

import runpy
import sys

import wearablepermed_utils.core as WPM_utils

""" 
# Extraer timestamp (columna 0) y calcular ENMO (columnas 1, 2, 3)
def calcular_enmo(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2) - 1

 """

def filtro_paso_bajo(senal, fs=30.0, fc=0.5, order=4):
    """
    Aplica un filtro Butterworth paso bajo a una señal.
    
    Parámetros:
    - senal: array 1D con los valores de la señal
    - fs: frecuencia de muestreo en Hz (por defecto 30 Hz para acelerómetros)
    - fc: frecuencia de corte en Hz (por defecto 0.5 Hz)
    - order: orden del filtro (por defecto 4)
    
    Retorna:
    - senal filtrada
    """
    if len(senal) < 2 * order:
        # Si la señal es muy corta, no filtrar
        return senal
    
    # Validar parámetros
    if fs <= 0 or fc <= 0:
        return senal
    
    # Diseño del filtro Butterworth
    nyquist = fs / 2.0
    
    # Asegurar que fc < nyquist (la frecuencia de corte debe ser menor que la frecuencia de Nyquist)
    if fc >= nyquist:
        # Si fc es muy alta, usar el 90% de la frecuencia de Nyquist
        fc = nyquist * 0.9
    
    normal_cutoff = fc / nyquist
    
    # Validar que normal_cutoff esté en el rango válido (0, 1)
    if normal_cutoff <= 0 or normal_cutoff >= 1:
        return senal
    
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Aplicar filtro (filtfilt para evitar desfase)
    senal_filtrada = signal.filtfilt(b, a, senal)
    
    return senal_filtrada

def plot_enmo_subplots(reference_signal, target_signal_original, target_signal_rescaled, rescaling_log_dir, plot_figures):
    """Plot three vertically stacked subplots (timestamp vs ENMO) and save the figure.
    argumetos: 

    """
    reference_enmo = WPM_utils.ENMO(reference_signal[:, 1:4])
    reference_enmo = np.abs(reference_enmo)
    reference_enmo_signal = np.column_stack([reference_signal[:, 0], reference_enmo])

    target_enmo_original = WPM_utils.ENMO(target_signal_original[:, 1:4])
    target_enmo_original = np.abs(target_enmo_original)
    target_enmo_original_signal = np.column_stack([target_signal_original[:, 0], target_enmo_original])

    target_enmo_rescaled = WPM_utils.ENMO(target_signal_rescaled[:, 1:4])
    target_enmo_rescaled = np.abs(target_enmo_rescaled)
    target_enmo_rescaled_signal = np.column_stack([target_signal_rescaled[:, 0], target_enmo_rescaled])


    if plot_figures:
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
        axes[0].plot(reference_enmo_signal[:, 0], reference_enmo_signal[:, 1], color='C0')
        axes[0].set_ylabel('REF ENMO')
        axes[0].set_title('PMP_REF - ENMO')
        axes[0].grid(True)

        axes[1].plot(target_enmo_original_signal[:, 0], target_enmo_original_signal[:, 1], color='C1')
        axes[1].set_ylabel('TGT ENMO ORIGINAL')
        axes[1].set_title('PMP_TARGET ORIGINAL - ENMO')
        axes[1].grid(True)

        axes[2].plot(target_enmo_rescaled_signal[:, 0], target_enmo_rescaled_signal[:, 1], color='C1')
        axes[2].set_ylabel('TGT ENMO REESCALADO')
        axes[2].set_title('PMP_TARGET REESCALADO - ENMO')
        axes[2].grid(True)

        # Plot each raw channel (columns 1..) in a different color and label
        axes[3].plot(target_signal_rescaled[:, 0], target_signal_rescaled[:, 1:4])
        axes[3].set_ylabel('TARGET RESCALED')
        axes[3].set_title('PMP_TARGET - RAW')
        axes[3].set_xlabel('timestamp')
        axes[3].legend(loc='upper right')
        axes[3].grid(True)

        fig.tight_layout()


        plt.show()


    # with open(f"{rescaling_log_dir}/rescaling_signals.pickle", "wb") as f:
    #     pickle.dump(fig, f)

    # fig.savefig(f"{rescaling_log_dir}/rescaling_signals.svg", format="svg", bbox_inches='tight')

    np.savez(
        f"{rescaling_log_dir}/enmo_signals.npz",
        reference_enmo_signal=reference_enmo_signal,
        target_enmo_original_signal=target_enmo_original_signal,
        target_enmo_rescaled_signal=target_enmo_rescaled_signal,
    )



def correlacion_con_timestamp(sig1, sig2, start_time=None, end_time=None,
                              first_hours=None, last_hours=None,
                              min_overlap_seconds=1.0, aplicar_filtro=True, fc=0.02):
    """
    Correlación Pearson entre dos señales con timestamps.

    Parámetros:
    - sig1, sig2: arrays Nx2 -> [timestamp, valor]
    - start_time, end_time: (opcional) recorte explícito en las mismas unidades temporales
      que los timestamps (por ejemplo segundos desde epoch). Si se proporcionan, se usan
      para limitar el intervalo de correlación.
    - first_hours: si se proporciona (float), se limita la correlación a las primeras
      `first_hours` horas del intervalo común.
    - last_hours: si se proporciona (float), se limita la correlación a las últimas
      `last_hours` horas del intervalo común.
    - min_overlap_seconds: solape mínimo aceptable (por defecto 1s). Si el solape es
      menor, devuelve np.nan.
    - aplicar_filtro: si es True, aplica un filtro paso bajo a las señales antes de correlacionar
    - fc: frecuencia de corte del filtro paso bajo en Hz (por defecto 0.5 Hz)

    Comportamiento:
    - Las opciones se combinan tomando la intersección del intervalo común con los
      recortes solicitados. first_hours/last_hours se expresan en horas.
    - La función interpola ambas series a una malla temporal común (dt = mediana de
      los dt disponibles) y calcula np.corrcoef sobre esa malla.
    - Si aplicar_filtro=True, se aplica un filtro paso bajo a ambas señales interpoladas.
    """

    # protección básica
    if sig1 is None or sig2 is None or len(sig1) == 0 or len(sig2) == 0:
        return np.nan

    t1, x1 = sig1[:, 0], sig1[:, 1]
    t2, x2 = sig2[:, 0], sig2[:, 1]

    # Intervalo temporal común inicial
    t_ini = max(t1[0], t2[0])
    t_fin = min(t1[-1], t2[-1])

    # Aplicar first_hours / last_hours si se piden
    if first_hours is not None:
        try:
            fh_sec = float(first_hours) * 3600.0
            t_fin = min(t_fin, t_ini + fh_sec)
        except Exception:
            pass

    if last_hours is not None:
        try:
            lh_sec = float(last_hours) * 3600.0
            t_ini = max(t_ini, t_fin - lh_sec)
        except Exception:
            pass

    # Aplicar start_time / end_time explícitos si los dieron (se asume misma escala)
    if start_time is not None:
        try:
            st = float(start_time)
            t_ini = max(t_ini, st)
        except Exception:
            pass
    if end_time is not None:
        try:
            et = float(end_time)
            t_fin = min(t_fin, et)
        except Exception:
            pass

    # Comprobar solape
    if t_fin <= t_ini + float(min_overlap_seconds):
        return np.nan

    # Base temporal común (usar la malla más densa disponible)
    # si alguno de los arrays tiene menos de 2 puntos, no podemos estimar dt
    if len(t1) < 2 or len(t2) < 2:
        return np.nan

    dt1 = np.median(np.diff(t1))
    dt2 = np.median(np.diff(t2))
    dt = min(dt1, dt2)
    if dt <= 0 or np.isnan(dt):
        return np.nan

    t_common = np.arange(t_ini, t_fin, dt)

    # Interpolación (si la malla común es vacía, devolvemos nan)
    if len(t_common) < 2:
        return np.nan

    x1_i = np.interp(t_common, t1, x1)
    x2_i = np.interp(t_common, t2, x2)

    # Aplicar filtro paso bajo si se solicita
    if aplicar_filtro:
        # Calcular frecuencia de muestreo a partir de dt
        # Los timestamps suelen estar en milisegundos, así que dt está en ms
        # fs = 1/dt donde dt está en segundos
        if dt > 100:  # Si dt > 100, probablemente está en milisegundos
            fs = 1000.0 / dt  # Convertir de ms a Hz
        else:
            fs = 1.0 / dt  # Ya está en segundos
        
        # Validar fs antes de filtrar
        if fs > 0 and fc < fs / 2.0:
            x1_i = filtro_paso_bajo(x1_i, fs=fs, fc=fc)
            x2_i = filtro_paso_bajo(x2_i, fs=fs, fc=fc)

    # Correlación de Pearson
    # si la varianza es 0 en alguna serie, corrcoef devuelve nan/inf; protegemos
    if np.nanstd(x1_i) == 0 or np.nanstd(x2_i) == 0:
        return np.nan

    return np.corrcoef(x1_i, x2_i)[0, 1]



##########################################
#######    1050_PI ########################
##########################################


def reescalar_senhal(dict, rescaling_log_dir, plot_figures):
    ref_file = dict['ref_file']
    segment_ref_file = dict['segment ref file']
    time_sincro_ref_file = datetime.strptime(dict['time_sincro_ref_file'], "%H:%M:%S").time() if dict.get('time_sincro_ref_file') else None
    sample_sincro_ref_file = dict['sample_sincro_ref_file']
    target_file = dict['target_file']
    activity_file = dict['activity_file']
    segment_target_file = dict['segment target file']
    sample_sincro_target_file = dict['sample_sincro_target_file']
    time_sincro_target_file = datetime.strptime(dict['time_sincro_target_file'], "%H:%M:%S").time() if dict.get('time_sincro_target_file') else None
    offset_range = dict['offset_range']
    offset_step = dict['offset_step']
    

    ref_scaled_data, dic_timing_ref = WPM_utils.load_scale_WPM_data(
        ref_file,
        segment_ref_file,
        activity_file,
        sample_sincro_ref_file,
        time_sincro_ref_file)

    tgt_scaled_data, dic_timing_tgt = WPM_utils.load_scale_WPM_data(
        target_file,
        segment_target_file,
        activity_file,
        sample_sincro_target_file,
        time_sincro_target_file)
    
    if time_sincro_ref_file is None:
        hora_fecha_ref, muestra_ref = busca_par_hora_fecha_actividad(ref_scaled_data[:,0], dic_timing_ref, Dia_1=False,
                                  primary_candidates='TROTAR - Hora de inicio',
                                  fallback_candidates='CAMINAR USUAL SPEED - Hora de inicio')
   
        time_sincro_ref_file = hora_fecha_ref.time()
        sample_sincro_ref_file = muestra_ref

    if time_sincro_target_file is None:
        hora_fecha_M, muestra_M = busca_par_hora_fecha_actividad(tgt_scaled_data[:,0], dic_timing_tgt, Dia_1=False,
                                  primary_candidates='TROTAR - Hora de inicio',
                                  fallback_candidates='CAMINAR USUAL SPEED - Hora de inicio')

        time_sincro_target_file = hora_fecha_M.time()
        sample_sincro_target_file = muestra_M

    best_signal_target_scaled_raw, correlations, best_K, best_offset = reescalar_senhal_2(ref_scaled_data, tgt_scaled_data, activity_file, segment_target_file, sample_sincro_target_file, time_sincro_target_file, offset_range, offset_step, rescaling_log_dir, plot_figures)
    plot_enmo_subplots(ref_scaled_data, tgt_scaled_data, best_signal_target_scaled_raw, rescaling_log_dir, plot_figures)


    return ref_scaled_data, best_signal_target_scaled_raw, correlations, best_K, best_offset


def reescalar_senhal_2(signal_ref_raw, signal_target_raw, activity_file, segment_body, sample_sincro, time_sample_sincro, offset_range = 100, offset_step = 1, rescaling_log_dir = ".", plot_figures = False):
    """
    a reescalar_senhal se le pueden pasara los datos cargados directamente (load_WPD_data) o los escalados con 
    una (muestra, hora) (load_scale_WPM_data). 

    No va a repercutir mas alla de el cálculo de las K. Si le pasamos los datos ya reescalados, para un 
    offset de 0 calculará una K de 1. En otro caso para un offset de 0 el valor de la K será el mismo que calcula
    la funcion de cargar y reescalar.

    En cualquier caso es necesario pasarle un par (muestra, hora) para que haga el reescalado moviendose en torno a esa referencia.print
    """
    
    #SEÑAL REFERENCIA [TIMESTAMP, ENMO]
    ref_timestamp = signal_ref_raw[:, 0]
    ref_enmo = WPM_utils.ENMO(signal_ref_raw[:, 1:4])
    signal_ref_enmo = np.column_stack([ref_timestamp, ref_enmo])

    #BUCLE DE OFFSETs
    sample_offset = np.arange(-offset_range, offset_range, offset_step)
    correlations = []
    Ks = []

    print("Calculando correlaciones para diferentes offsets...")
    print("----------------------------------------")

    # Queremos maximizar la correlación Pearson (r en [-1,1]). Inicializar a -inf
    # permite que cualquier valor numérico mayor lo reemplace.
    best_corre = -np.inf
    best_offset = np.nan
    best_K=1.0

    for offset in sample_offset:

        #señal target reescalada K segun muestra de offset actual
        K = WPM_utils.calculate_accelerometer_drift(signal_target_raw, activity_file, segment_body, sample_sincro + offset, time_sample_sincro) 
        signal_target_scaled_raw = WPM_utils.apply_scaling_to_matrix_data(signal_target_raw, K)

        # SEÑAL REESCALADA [TIMESTAMP, ENMO]
        target_scaled_enmo = WPM_utils.ENMO(signal_target_scaled_raw[:, 1:4])
        signal_target_scaled_enmo = np.column_stack([signal_target_scaled_raw[:, 0], target_scaled_enmo])

        # Calcular correlación dos ultimas horas
        corre = correlacion_con_timestamp(signal_ref_enmo, signal_target_scaled_enmo, last_hours=2.0)

        # Buscamos la correlación máxima
        if corre > best_corre:
            best_corre = corre
            best_K = K
            best_offset = offset
            best_signal_target_scaled_raw = signal_target_scaled_raw.copy()

        print(f"Offset: {offset} -> Correlación: {corre:.4f}")
        correlations.append(corre)
        Ks.append(K)

    correlations = np.array(correlations)


    fig, ax = plt.subplots()

    ax.plot(sample_offset, correlations)
    ax.axvline(x=best_offset, linestyle='--', label=f'Best offset = {best_offset}')

    ax.set_title('Correlaciones para diferentes offsets')
    ax.set_xlabel('Offset index')
    ax.set_ylabel('Correlación Pearson')
    ax.legend()
    ax.grid(True)

    fig.savefig(f"{rescaling_log_dir}/correlaciones.svg", format="svg", bbox_inches='tight')

    if plot_figures:    
        plt.show()
    else:
        plt.close(fig)

    return best_signal_target_scaled_raw, correlations, best_K, best_offset


def save_reescale_log(best_k, best_offset, correlations, inputs_dict=None, log_dir=None):
    """
    Guarda un log con los resultados del reescalado incluyendo los parámetros de entrada.
    """
    if log_dir is None:
        log_dir = "."

    log_path = os.path.join(log_dir, "numerical_log.log")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "best_K": best_k,
        "best_offset": int(best_offset),
        "correlations": [float(c) if not np.isnan(c) else None for c in correlations]
    }
    
    if inputs_dict is not None:
        log_entry["inputs"] = inputs_dict

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry, indent=2, default=str) + "\n")

#######################################
######################################


def find_activity_start_date_time(dic_timing, Dia_1=False,
                                  primary_candidates='TROTAR - Hora de inicio',
                                  fallback_candidates='CAMINAR USUAL SPEED - Hora de inicio'):
    """
    Buscar la hora de inicio de la actividad principal (JOGGIN/TROTAR) y, si no existe,
    devolver la hora de inicio de WALKING USUAL SPEED (CAMINAR USUAL SPEED).

    La búsqueda es case-insensitive y hace coincidencia por substring en las claves
    del diccionario `dic_timing`. Devuelve el valor encontrado o None si no hay nada.
    """
    if not dic_timing:
        return None

    # normalize dictionary keys for case-insensitive substring search
    # remove diacritics to avoid mismatches like 'día' vs 'dia'
    def _norm(s):
        if s is None:
            return None
        s2 = str(s)
        s2 = unicodedata.normalize('NFKD', s2)
        s2 = ''.join(ch for ch in s2 if not unicodedata.combining(ch))
        s2 = ' '.join(s2.split())
        return s2.lower()

    key_map = {_norm(k): v for k, v in dic_timing.items()}

    # prepare candidate lists
    # normalize candidate strings similarly
    if isinstance(primary_candidates, (list, tuple)):
        prim_list = [_norm(p) for p in primary_candidates]
    else:
        prim_list = [_norm(primary_candidates)]

    if isinstance(fallback_candidates, (list, tuple)):
        fall_list = [_norm(p) for p in fallback_candidates]
    else:
        fall_list = [_norm(fallback_candidates)]

    def search_candidates(candidate_list):
        for cand in candidate_list:
            for k, v in key_map.items():
                if cand in k and v is not None:
                    return v
        return None

    # find the activity time (primary -> fallback)
    act_time = search_candidates(prim_list)
    if act_time is None:
        act_time = search_candidates(fall_list)
    if act_time is None:
        return None

    # determine which date to use (Fecha día 1 or Fecha día 7)
    if Dia_1:
        date_keys = [_norm('Fecha día 1')]
    else:
        date_keys = [_norm('Fecha día 7')]

    activity_date = None
    for dk in date_keys:
        if dk in key_map and key_map[dk] is not None:
            activity_date = key_map[dk]
            break


    # act_time is typically a datetime.time; if we have a date, combine
    try:
        if activity_date is not None:
            # If activity_date is a date-like object, combine with time
            return datetime.combine(activity_date, act_time)
    except Exception:
        # fall through and return the raw time if combine fails
        pass

    return act_time


#SECCION CON ARCHIVO SIN NECESIDAD DE CORRECCION
# El 1049 en principio esta todo correcto.

def busca_par_hora_fecha_actividad(ts_array, dic_timing, Dia_1=False,
                                  primary_candidates='TROTAR - Hora de inicio',
                                  fallback_candidates='CAMINAR USUAL SPEED - Hora de inicio'):
    """
    Busca un par de hora y fecha de actividad en el diccionario de tiempos.
    """
    hora_fecha = find_activity_start_date_time(dic_timing)
    
    hora_fecha_ms = hora_fecha.timestamp() * 1000
    muestra = WPM_utils.find_closest_timestamp(ts_array, hora_fecha_ms)
    print(hora_fecha)

    return hora_fecha, muestra



def reescala(base_dir, sujeto, segmento_ref, segmento_target, time_sincro_ref_file, sample_sincro_ref_file, time_sincro_target_file, sample_sincro_target_file, offset_range, step_range, plot_figures = False):
        
    inputs = {
            'ref_file': f"{base_dir}/{sujeto}/{sujeto}_W1_{segmento_ref}.csv",
            'segment ref file': segmento_ref,
            'time_sincro_ref_file': time_sincro_ref_file,
            'sample_sincro_ref_file': sample_sincro_ref_file,
            'target_file': f"{base_dir}/{sujeto}/{sujeto}_W1_{segmento_target}.csv",
            'activity_file': f"{base_dir}/{sujeto}/{sujeto}_RegistroActividades.xlsx",
            'segment target file': segmento_target,
            'sample_sincro_target_file': sample_sincro_target_file,
            'time_sincro_target_file': time_sincro_target_file,
            'offset_range': offset_range,
            'offset_step': step_range,   
        }
    
    log_dir = f"{base_dir}/{sujeto}/rescaling_log_{segmento_target}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Created or exists: {log_dir}")

    rescaled_ref, rescaled_target, correlations, best_K, best_offset = reescalar_senhal(inputs, log_dir, plot_figures)
    print("#----------------------------------------#")
    print(f"Mejor K para {sujeto}_{segmento_target}:", best_K)
    print(f"Mejor offset para {sujeto}_{segmento_target}:", best_offset)
    save_reescale_log(best_K, best_offset, correlations, inputs, log_dir)
    np.savez_compressed(f"{base_dir}/{sujeto}/{sujeto}_W1_{segmento_ref}_rescaled.npz", datos=rescaled_ref)
    np.savez_compressed(f"{base_dir}/{sujeto}/{sujeto}_W1_{segmento_target}_rescaled.npz", datos=rescaled_target)
    print("=========================")    
    print("\n\n")


if __name__ == "__main__":
    # reescala("./data", "PMP1050", 
    #          'PI', 'M', 
    #          "12:34:11", 12012445, 
    #          "12:44:20", 13378200, 
    #          100, 40,
    #          False)
#     reescala("./data", "PMP1050", 
#              'PI', 'C', 
#              "12:34:11", 12012445, 
#              "12:34:11", 12591189, 
#              100, 1,
#              False)

    
# ###

#     reescala("./data", "PMP1049", 
#              'PI', 'M', 
#              None, None, 
#              None, None, 
#              100, 1,
#              False)
#     reescala("./data", "PMP1049", 
#              'PI', 'C', 
#              None, None, 
#              None, None, 
#              100, 1,
#              False)
    pass