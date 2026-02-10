import pandas as pd

df = pd.read_excel(
    "Final_Resume_v3_Alejandro_sincronizacion.xlsx",
    sheet_name="End date time problem(E4)"
)

df = df.rename(columns={
    "Acceloremeters": "Accelerometer",
    "Sample Numbers": "Sample Number"
})

with open("sync_thigh_to_wrist.py", "w") as f:
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

        line = (
            f'reescala(base_dir, "{participant}", segmento_ref, segmento_target, '
            f'"{h["Excel Hour"]}", {h["Sample Number"]}, '
            f'"{w["Excel Hour"]}", {w["Sample Number"]}, '
            f'offset_range, step_range)'
        )
        f.write(line + "\n")
