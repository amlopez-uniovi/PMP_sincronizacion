import pandas as pd
import datetime

df = pd.read_excel(
    "Final_Resume_v3_Alejandro_sincronizacion.xlsx",
    sheet_name="End date time problem(E4)"
)

df = df.rename(columns={
    "Acceloremeters": "Accelerometer",
    "Sample Numbers": "Sample Number"
})

with open("script_conversion.py", "w") as f:
    # Cabecera con variables usadas por los scripts generados
    f.write("base_dir = './Input'\n")
    f.write("segmento_ref = 'PI'\n")
    f.write("segmento_target = 'M'\n")
    f.write("offset_range = 125\n")
    f.write("step_range = 1\n\n")

    f.write("from pmp_sincro import reescala\n\n")

    # Lista completa de participantes que queremos procesar; si no aparecen
    # en el Excel se generará una llamada a reescala con None para horas y muestras
    required_participants = [
        'PMP1011','PMP1013','PMP1017','PMP1019','PMP1020','PMP1021','PMP1022',
        'PMP1024','PMP1025','PMP1026','PMP1027','PMP1029','PMP1030','PMP1032',
        'PMP1036','PMP1038','PMP1039','PMP1043','PMP1046','PMP1047','PMP1048',
        'PMP1049','PMP1050','PMP1051','PMP1052','PMP1053','PMP1055','PMP1056',
        'PMP1057','PMP1058','PMP1059','PMP1060','PMP1061','PMP1062','PMP1063',
        'PMP1064','PMP1065','PMP1066','PMP1067','PMP1068','PMP1070','PMP1071',
        'PMP1072','PMP1074','PMP1075','PMP1076','PMP1077','PMP1079','PMP1080',
        'PMP1084','PMP1085','PMP1087','PMP1088','PMP1090','PMP1091','PMP1092',
        'PMP1093','PMP1094','PMP1095','PMP1097'
    ]

    processed = set()
    for participant, g in df.groupby("Participant"):
        # Verificar que tenga al menos "thigh" y "wrist"
        accelerometers = g["Accelerometer"].str.lower().unique()
        
        # Solo procesar si tiene "thigh" y "wrist" (puede tener otros como "hip")
        if not {"thigh", "wrist"}.issubset(set(accelerometers)):
            continue
        
        thigh = g[g["Accelerometer"].str.lower() == "thigh"]
        wrist = g[g["Accelerometer"].str.lower() == "wrist"]

        h = thigh.iloc[0]
        w = wrist.iloc[0]

        # Marcar como procesado
        processed.add(participant)

        # Obtener valores; si faltan las horas, NO generamos la llamada a
        # reescala (opción 1). Escribimos un comentario para revisión manual.
        raw_h_hour = h.get("Excel Hour") if pd.notna(h.get("Excel Hour")) else None
        h_sample = int(h.get("Sample Number")) if pd.notna(h.get("Sample Number")) else None
        raw_w_hour = w.get("Excel Hour") if pd.notna(w.get("Excel Hour")) else None
        w_sample = int(w.get("Sample Number")) if pd.notna(w.get("Sample Number")) else None

        # Forzar conversión de las horas a cadena (formato HH:MM:SS) para
        # garantizar que en el archivo generado aparezcan entre comillas.
        def to_hour_str(v):
            if v is None:
                return None
            try:
                if isinstance(v, pd.Timestamp) or isinstance(v, datetime.datetime) or isinstance(v, datetime.time):
                    return v.strftime('%H:%M:%S')
            except Exception:
                pass
            # Si viene como número u otro tipo, convertir a str
            return str(v)

        h_hour = to_hour_str(raw_h_hour)
        w_hour = to_hour_str(raw_w_hour)

        # Si falta cualquiera de las horas, no generar la llamada y añadir
        # un comentario indicando qué falta para este participante.
        missing_parts = []
        if h_hour is None:
            missing_parts.append('thigh hour')
        if w_hour is None:
            missing_parts.append('wrist hour')

        if missing_parts:
            f.write(f'# OMITTED: {participant} - falta: {", ".join(missing_parts)}; revisar Excel\n')
            continue

        # Forzar que las horas en la llamada queden entre comillas dobles
        h_hour_code = f'"{h_hour}"'
        w_hour_code = f'"{w_hour}"'

        # Números: si sample es None (raro) lo dejamos como None literal
        h_sample_code = str(h_sample) if h_sample is not None else 'None'
        w_sample_code = str(w_sample) if w_sample is not None else 'None'

        line = (
            f'reescala(base_dir, "{participant}", segmento_ref, segmento_target, '
            f'{h_hour_code}, {h_sample_code}, '
            f'{w_hour_code}, {w_sample_code}, '
            f'offset_range, step_range)'
        )
        f.write(line + "\n")

    # Generar llamadas para participantes requeridos que no estaban en el Excel
    missing = [p for p in required_participants if p not in processed]
    if missing:
        f.write("\n# Participantes sin entrada en el Excel: llamadas con None para horas y muestras\n")
        for participant in missing:
            line = (
                f'reescala(base_dir, "{participant}", segmento_ref, segmento_target, '
                f'None, None, None, None, offset_range, step_range)'
            )
            f.write(line + "\n")


