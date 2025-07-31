import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.utils.multiclass import type_of_target

pd.set_option("display.float_format", "{:.6f}".format)
import os
import platform
import subprocess
import numpy as np

from src.data import load_csv, save_csv
from src.preprocess import preprocess
from src.model import train_model
from src.evaluate import evaluate
from src.visualize import plot_target_distribution, plot_predictions
from src.crossvalidate import cross_validate_model
import os


class DataMiningUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Mining Project - by TEAM A")
        self.root.geometry("700x650")
        self.df = None
        self.model = None
        self.y_test = None
        self.y_pred = None
        self.dataset_path = ttk.StringVar()

        self.target_column = ttk.StringVar()
        self.target_column.set("")  # Khởi tạo trống

        # Title label
        ttk.Label(root, text="Data Mining UI", font=("Segoe UI", 22, "bold")).pack(
            pady=20
        )

        # Dataset selection
        ttk.Label(root, text="Selected Dataset:", font=("Segoe UI", 12)).pack(
            pady=(0, 5)
        )
        ttk.Entry(
            root,
            textvariable=self.dataset_path,
            width=70,
            font=("Segoe UI", 11),
            state="readonly",
        ).pack(pady=5)
        ttk.Button(
            root,
            text="Choose Dataset",
            bootstyle="info-outline",
            width=20,
            command=self.choose_dataset,
        ).pack(pady=10)

        # Action buttons
        ttk.Button(
            root,
            text="Load Data",
            bootstyle="primary",
            width=40,
            command=self.load_data,
        ).pack(pady=5)
        ttk.Button(
            root,
            text="Explore Data",
            bootstyle="info",
            width=40,
            command=self.explore_data,
        ).pack(pady=5)
        ttk.Button(
            root,
            text="Preprocess Data",
            bootstyle="secondary",
            width=40,
            command=self.preprocess_data,
        ).pack(pady=5)
        ttk.Button(
            root,
            text="Train Model",
            bootstyle="success",
            width=40,
            command=self.train_model,
        ).pack(pady=5)
        ttk.Button(
            root,
            text="Evaluate Model",
            bootstyle="warning",
            width=40,
            command=self.evaluate_model,
        ).pack(pady=5)
        ttk.Button(
            root,
            text="Cross Validate",
            bootstyle="light",
            width=40,
            command=self.cross_validate_model,
        ).pack(pady=5)

        ttk.Button(
            root, text="Visualize", bootstyle="danger", width=40, command=self.visualize
        ).pack(pady=5)

        # Debug log panel
        ttk.Label(root, text="DEBUG LOG:", font=("Segoe UI", 11, "bold")).pack(
            pady=(15, 10)
        )
        self.log_text = ttk.Text(root, height=10, font=("Consolas", 10))
        self.log_text.pack(fill=BOTH, expand=True, padx=15, pady=5)

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def choose_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset CSV",
            filetypes=[("CSV Files", "*.csv")],
            initialdir=os.path.join(os.getcwd(), "data/raw"),
        )
        if file_path:
            self.dataset_path.set(file_path)
            self.log(f"[INFO] Selected file: {file_path}")

    def load_data(self):
        path = self.dataset_path.get()
        if not path:
            messagebox.showerror("Error", "Please choose a dataset file first.")
            return
        self.df = load_csv(path)
        self.log(f"[INFO] Loaded dataset with shape: {self.df.shape}")
        messagebox.showinfo("Success", f"Loaded data with {self.df.shape[0]} rows.")

    def explore_data(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        info = ""
        info += f"🔹 Shape: {self.df.shape}\n\n"
        info += "🔹 Data Types:\n" + str(self.df.dtypes) + "\n\n"
        info += "🔹 Missing Values:\n" + str(self.df.isnull().sum()) + "\n\n"
        info += f"🔹 Duplicate Rows: {self.df.duplicated().sum()}\n\n"

        self.log("[INFO] Exploring data...")
        self.log(f"Shape: {self.df.shape}")
        self.log("Data types:\n" + str(self.df.dtypes))
        self.log("Missing values:\n" + str(self.df.isnull().sum()))
        self.log("Duplicate rows: " + str(self.df.duplicated().sum()))
        try:
            info += (
                "🔹 Numerical Description:\n" + self.df.describe().to_string() + "\n\n"
            )
        except:
            info += "🔹 No numeric description available.\n\n"

        popup = ttk.Toplevel(self.root)
        popup.title("Explore Data Summary")
        popup.geometry("1200x600")

        ttk.Label(
            popup, text="Data Exploration Summary", font=("Segoe UI", 14, "bold")
        ).pack(pady=10)

        text_widget = ttk.Text(popup, wrap="word", font=("Consolas", 14))
        text_widget.insert("1.0", info)
        text_widget.config(state="disabled")
        text_widget.pack(fill="both", expand=True, padx=15, pady=10)

        self.log("[INFO] Exploration summary shown in popup window.")

    def preprocess_data(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        self.log("[INFO] Starting preprocessing...")
        self.X_train, self.X_test, self.y_train, self.y_test, self.df = preprocess(
            self.df
        )

        input_filename = os.path.basename(self.dataset_path.get())  # "2019.csv"
        output_filename = f"preprocessed_{input_filename}"  # "processed_2019.csv"

        save_csv(self.df, output_filename)
        self.log(
            f"[INFO] Preprocessing complete. Saved to data/preprocessed/{output_filename}"
        )
        messagebox.showinfo(
            "Done", f"Preprocessing complete and saved to {output_filename}."
        )
        try:
            processed_path = os.path.abspath(
                os.path.join("data", "processed", output_filename)
            )
            if platform.system() == "Windows":
                os.startfile(processed_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", processed_path])
            else:  # Linux
                subprocess.run(["xdg-open", processed_path])
        except Exception as e:
            self.log(f"[ERROR] Failed to open file: {e}")

        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        ttk.Label(self.root, text="Select Target Column:", font=("Segoe UI", 11)).pack()
        ttk.Combobox(
            self.root,
            textvariable=self.target_column,
            values=numeric_cols,
            state="readonly",
        ).pack()

    def train_model(self):
        if not hasattr(self, "X_train") or self.df is None:
            messagebox.showerror("Error", "Please preprocess the data first.")
            return

        def proceed_training(selected_target):
            self.target_column.set(selected_target)
            try:
                # Lấy lại y_train và y_test từ df theo chỉ mục
                self.y_train = self.df.loc[self.X_train.index, selected_target]
                self.y_test = self.df.loc[self.X_test.index, selected_target]

                task_type = type_of_target(self.y_train)
                task = (
                    "classification"
                    if task_type in ["binary", "multiclass"]
                    else "regression"
                )

                # Train
                self.model, self.y_pred = train_model(
                    self.X_train, self.y_train, self.X_test, task_type=task
                )

                # Log
                y_test_preview = self.y_test[:5].tolist()
                y_pred_preview = self.y_pred[:5].tolist()
                self.log(f"[DEBUG] y_test (first 5): {y_test_preview}")
                self.log(f"[DEBUG] y_pred (first 5): {y_pred_preview}")

                # Visualize
                plot_predictions(
                    self.y_test, self.y_pred, target_name=self.target_column.get()
                )

            except Exception as e:
                self.log(f"[ERROR] Training failed: {e}")
                messagebox.showerror("Error", f"Training failed: {e}")

        # Nếu chưa có target_column hiện popup
        if not hasattr(self, "target_column"):
            self.target_column = ttk.StringVar()

        popup = ttk.Toplevel(self.root)
        popup.title("Select Target Column")
        popup.geometry("400x200")

        ttk.Label(popup, text="Select Target Column:", font=("Segoe UI", 12)).pack(
            pady=15
        )

        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        dropdown = ttk.Combobox(
            popup,
            values=numeric_cols,
            textvariable=self.target_column,
            state="readonly",
        )
        dropdown.pack(pady=10)

        ttk.Button(
            popup,
            text="Train Now",
            bootstyle="success",
            command=lambda: proceed_training(self.target_column.get()),
        ).pack(pady=10)

    def evaluate_model(self):
        if self.df is None or self.y_pred is None or self.X_test is None:
            messagebox.showerror("Error", "You must train the model first.")
            return

        target = self.target_column.get()
        if not target:
            messagebox.showerror("Error", "No target column was trained.")
            return

        popup = ttk.Toplevel(self.root)
        popup.title(f"Evaluate Model – Target: {target}")
        popup.geometry("500x420")

        ttk.Label(
            popup,
            text=f"Evaluating target: {target} - The last trained",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(15, 5))
        ttk.Label(popup, text="Select Task Type:", font=("Segoe UI", 11)).pack(
            pady=(5, 2)
        )

        task_var = ttk.StringVar(value="regression")
        task_menu = ttk.Combobox(
            popup, textvariable=task_var, state="readonly", width=30
        )
        task_menu["values"] = ["regression", "classification", "clustering"]
        task_menu.pack(pady=5)

        result_text = ttk.Text(popup, height=10, font=("Consolas", 14), wrap="word")
        result_text.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        result_text.config(state="disabled")

        def evaluate_now():
            task_type = task_var.get()
            try:
                import numpy as np
                from sklearn.utils.multiclass import type_of_target

                y_true = self.df.loc[self.X_test.index, target]
                y_pred = self.y_pred

                if task_type == "classification":
                    if type_of_target(y_true) == "continuous":
                        self.log("[WARN] y_test is continuous. Rounding.")
                        y_true = np.round(y_true).astype(int)
                    if type_of_target(y_pred) == "continuous":
                        self.log("[WARN] y_pred is continuous. Rounding.")
                        y_pred = np.round(y_pred).astype(int)

                results = evaluate(
                    y_true, y_pred, task_type=task_type, X_test=self.X_test
                )
                result_msg = "\n".join(
                    [
                        (
                            f"{metric}: {value:.4f}"
                            if isinstance(value, (int, float))
                            else f"{metric}: {value}"
                        )
                        for metric, value in results.items()
                    ]
                )

                self.log(
                    f"[INFO] Evaluation Results for {task_type} - Target: {target}"
                )
                result_text.config(state="normal")
                result_text.delete("1.0", "end")
                result_text.insert("1.0", result_msg)
                result_text.config(state="disabled")

            except Exception as e:
                self.log(f"[ERROR] Evaluation failed: {e}")
                messagebox.showerror("Error", f"Evaluation failed: {e}")

        ttk.Button(
            popup, text="Evaluate Now", bootstyle="success", command=evaluate_now
        ).pack(pady=5)

    def cross_validate_model(self):
        if self.df is None or not self.target_column.get():
            messagebox.showerror(
                "Error", "Please preprocess the data and select a target column."
            )
            return

        target = self.target_column.get()

        popup = ttk.Toplevel(self.root)
        popup.title(f"Cross Validation – Target: {target}")
        popup.geometry("500x420")

        ttk.Label(
            popup,
            text=f"Cross-validating target: {target}",
            font=("Segoe UI", 12, "bold"),
        ).pack(pady=(15, 5))

        ttk.Label(popup, text="Select Task Type:", font=("Segoe UI", 11)).pack(
            pady=(5, 2)
        )

        task_var = ttk.StringVar(value="regression")
        task_menu = ttk.Combobox(
            popup, textvariable=task_var, state="readonly", width=30
        )
        task_menu["values"] = ["regression", "classification"]
        task_menu.pack(pady=5)

        result_text = ttk.Text(popup, height=10, font=("Consolas", 14), wrap="word")
        result_text.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        result_text.config(state="disabled")

        def cross_validate_now():
            task_type = task_var.get()
            try:

                y = self.df[target]
                X = self.df.drop(columns=[target])

                # Kiểm tra nếu chọn classification mà target column là continuous
                target_type = type_of_target(y)
                if task_type == "classification" and target_type == "continuous":
                    warning_msg = (
                        f"Target '{target}' is a continuous variable and cannot be used "
                        f"for classification.\nPlease choose 'regression' instead."
                    )
                    self.log(f"[WARN] {warning_msg}")
                    result_text.config(state="normal")
                    result_text.delete("1.0", "end")
                    result_text.insert("1.0", warning_msg)
                    result_text.config(state="disabled")
                    return

                scores = cross_validate_model(X, y, task_type=task_type, cv=5)

                result_msg = "\n".join(
                    [f"Fold {i+1}: {score:.4f}" for i, score in enumerate(scores)]
                )
                result_msg += f"\n\nAverage Score: {np.mean(scores):.4f}"

                self.log(
                    f"[INFO] Cross-validation results for {task_type} - Target: {target}"
                )
                result_text.config(state="normal")
                result_text.delete("1.0", "end")
                result_text.insert("1.0", result_msg)
                result_text.config(state="disabled")

            except Exception as e:
                self.log(f"[ERROR] Cross-validation failed: {e}")
                messagebox.showerror("Error", f"Cross-validation failed: {e}")

        ttk.Button(
            popup,
            text="Cross Validate Now",
            bootstyle="info",
            command=cross_validate_now,
        ).pack(pady=5)

    def visualize(self):
        if self.df is None or not hasattr(self, "target_column") or self.y_pred is None:
            messagebox.showerror("Error", "Please load data and train model first.")
            return

        try:
            target = self.target_column.get()

            plot_target_distribution(self.df, target)

        except Exception as e:
            self.log(f"[ERROR] Visualization failed: {e}")
            messagebox.showerror("Error", f"Visualization failed: {e}")


if __name__ == "__main__":
    app = ttk.Window(
        themename="solar"
    )  #'cyborg', 'darkly', 'journal', 'flatly', 'solar'
    DataMiningUI(app)
    app.mainloop()
