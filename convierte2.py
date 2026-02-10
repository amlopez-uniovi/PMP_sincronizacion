subjects = [
    'PMP1011', 'PMP1022', 'PMP1046', 'PMP1047', 'PMP1049',
    'PMP1051', 'PMP1053', 'PMP1055', 'PMP1056', 'PMP1057',
    'PMP1058', 'PMP1059', 'PMP1060', 'PMP1061', 'PMP1062',
    'PMP1063', 'PMP1064', 'PMP1065', 'PMP1066', 'PMP1067',
    'PMP1068', 'PMP1070', 'PMP1071', 'PMP1074', 'PMP1076',
    'PMP1077', 'PMP1079', 'PMP1080', 'PMP1084', 'PMP1085',
    'PMP1087', 'PMP1088', 'PMP1090', 'PMP1091', 'PMP1092',
    'PMP1095', 'PMP1097'
]

output_file = "sync_thigh_to_wrist_2.py"

with open(output_file, "w") as f:
    f.write("# Auto-generated reescala calls\n\n")
    for sujeto in subjects:
        f.write(
            f'reescala(base_dir, "{sujeto}", segmento_ref, segmento_target, '
            f'None, None, None, None, offset_range, step_range)\n'
        )

print(f"Archivo generado: {output_file}")
