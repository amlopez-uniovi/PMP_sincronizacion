import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

# Matplotlib + TkAgg integration
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def abrir_archivo():
    file_path = filedialog.askopenfilename(
        filetypes=[("NumPy NPZ", "*.npz"), ("All files", "*.*")]
    )
    if not file_path:
        return

    try:
        data = np.load(file_path)

        # --- validar claves esperadas ---
        required_keys = {
            "reference_enmo_signal",
            "target_enmo_original_signal",
            "target_enmo_rescaled_signal",
        }

        if not required_keys.issubset(data.files):
            raise ValueError(
                f"El archivo no contiene las claves esperadas.\n"
                f"Esperadas: {required_keys}\n"
                f"Encontradas: {set(data.files)}"
            )

        reference = data["reference_enmo_signal"]
        target_orig = data["target_enmo_original_signal"]
        target_rescaled = data["target_enmo_rescaled_signal"]

        # --- recrear figura ---
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

        axes[0].plot(reference[:, 0], reference[:, 1], color="C0")
        axes[0].set_title("Reference ENMO")

        axes[1].plot(target_orig[:, 0], target_orig[:, 1], color="C2")
        axes[1].set_title("Target ENMO (original)")

        axes[2].plot(
            target_rescaled[:, 0],
            target_rescaled[:, 1],
            color="C1"
        )
        axes[2].set_title("Target ENMO (rescaled)")

        for ax in axes:
            ax.grid(True)

        fig.tight_layout()

        # --- mostrar en ventana Tk ---
        top = tk.Toplevel(root)
        top.title(f"ENMO viewer - {file_path}")

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo leer el archivo:\n{e}")


# ---------------- UI ----------------

root = tk.Tk()
root.title("ENMO NPZ Viewer")
root.geometry("600x400")

btn = tk.Button(root, text="Abrir archivo NPZ", command=abrir_archivo)
btn.pack(pady=10)

info = tk.Label(
    root,
    text="Carga archivos .npz y recrea la figura ENMO",
    fg="gray"
)
info.pack(pady=5)

root.mainloop()
