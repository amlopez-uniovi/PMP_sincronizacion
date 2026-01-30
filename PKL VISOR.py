import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
import pprint

# Matplotlib + TkAgg integration to show pickled figures
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def abrir_archivo():
    # Accept both .pkl and .pickle; use a tuple of patterns so the filedialog
    # recognizes both on all platforms (semicolon-separated strings can fail).
    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle Files", ("*.pkl", "*.pickle")), ("All files", "*.*")]
    )
    if file_path:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # If the object is a matplotlib Figure (or contains one), show it
                def _find_figure(obj):
                    # direct Figure
                    if isinstance(obj, Figure):
                        return obj
                    # dict-like: check common keys
                    if isinstance(obj, dict):
                        for k in ('fig', 'figure'):
                            if k in obj and isinstance(obj[k], Figure):
                                return obj[k]
                        # try to find any Figure value
                        for v in obj.values():
                            if isinstance(v, Figure):
                                return v
                    # objects that hold figure attribute
                    if hasattr(obj, 'figure') and isinstance(getattr(obj, 'figure'), Figure):
                        return getattr(obj, 'figure')
                    return None

                fig = _find_figure(data)
                if fig is not None:
                    # Create a new top-level window to display the figure
                    top = tk.Toplevel(root)
                    top.title(f"Figure viewer - {file_path}")
                    canvas = FigureCanvasTkAgg(fig, master=top)
                    canvas.draw()
                    widget = canvas.get_tk_widget()
                    widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                    toolbar = NavigationToolbar2Tk(canvas, top)
                    toolbar.update()
                    toolbar.pack(side=tk.TOP, fill=tk.X)
                else:
                    text_area.delete('1.0', tk.END)
                    text_area.insert(tk.END, pprint.pformat(data))
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el archivo: {e}")

# Interfaz mínima
root = tk.Tk()
root.title("MinPickle Viewer")
root.geometry("600x400")

btn = tk.Button(root, text="Abrir Archivo Pickle", command=abrir_archivo)
btn.pack(pady=10)

text_area = tk.Text(root, wrap=tk.WORD)
text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

root.mainloop()
