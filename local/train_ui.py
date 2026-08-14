import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from trainer import TrainerEngine

class TrainUI:

    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        self.root.after(
            100,
            self.process_log_queue
        )
        self.root.title("AI增量训练工具")
        self.root.geometry("1024x768")
        self.root.minsize(900, 650)
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)
        # title
        title_label = ttk.Label(
            main_frame,
            text="AI增量训练工具",
            font=("Microsoft YaHei", 18, "bold"),
            anchor="center"
        )
        title_label.pack(fill="x", pady=(0, 15))
        # model frame
        model_frame = ttk.LabelFrame(
            main_frame,
            text="基准模型",
            padding=10
        )
        model_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(
            model_frame,
            text="模型路径："
        ).grid(row=0, column=0, padx=(0, 8), pady=5)
        self.model_path = tk.StringVar()
        model_entry = ttk.Entry(
            model_frame,
            textvariable=self.model_path
        )
        model_entry.grid(
            row=0,
            column=1,
            sticky=tk.EW,
            pady=5
        )
        model_button = ttk.Button(
            model_frame,
            text="选择...",
            width=10,
            command=self.select_model
        )
        model_button.grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )
        model_frame.columnconfigure(1, weight=1)
        # train data
        dataset_frame = ttk.LabelFrame(
            main_frame,
            text="训练数据",
            padding=10
        )
        dataset_frame.pack(
            fill="x",
            pady=(0, 10)
        )
        ttk.Label(
            dataset_frame,
            text="数据文件："
        ).grid(
            row=0,
            column=0,
            padx=(0, 8),
            pady=5
        )
        self.dataset_path = tk.StringVar()
        ttk.Entry(
            dataset_frame,
            textvariable=self.dataset_path
        ).grid(
            row=0,
            column=1,
            sticky=tk.EW,
            pady=5
        )
        ttk.Button(
            dataset_frame,
            text="选择...",
            width=10,
            command=self.select_dataset
        ).grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )
        dataset_frame.columnconfigure(1, weight=1)
        # output
        output_frame = ttk.LabelFrame(
            main_frame,
            text="输出设置",
            padding=10
        )
        output_frame.pack(
            fill="x",
            pady=(0, 10)
        )
        ttk.Label(
            output_frame,
            text="输出目录："
        ).grid(
            row=0,
            column=0,
            padx=(0, 8),
            pady=5
        )
        # 默认输出目录
        self.output_path = tk.StringVar(
            value=r"D:\trained"
        )
        ttk.Entry(
            output_frame,
            textvariable=self.output_path
        ).grid(
            row=0,
            column=1,
            sticky=tk.EW,
            pady=5
        )
        ttk.Button(
            output_frame,
            text="选择...",
            width=10,
            command=self.select_output
        ).grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )
        output_frame.columnconfigure(
            1,
            weight=1
        )
        # Training Parameters
        parameter_frame = ttk.LabelFrame(
            main_frame,
            text="训练参数",
            padding=10
        )
        parameter_frame.pack(
            fill="x",
            pady=(0, 10)
        )
        # -----------------------------------------------------
        # CPU / GPU
        # -----------------------------------------------------
        ttk.Label(
            parameter_frame,
            text="训练设备："
        ).grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=5,
            pady=5
        )
        self.device = tk.StringVar(
            value="GPU"
        )
        self.device_combo = ttk.Combobox(
            parameter_frame,
            textvariable=self.device,
            values=["GPU", "CPU"],
            state="readonly",
            width=12
        )
        self.device_combo.grid(
            row=0,
            column=1,
            sticky=tk.W,
            padx=5,
            pady=5
        )
        # -----------------------------------------------------
        # Epoch
        # -----------------------------------------------------
        ttk.Label(
            parameter_frame,
            text="训练轮数："
        ).grid(
            row=1,
            column=0,
            sticky=tk.W,
            padx=5,
            pady=5
        )
        self.epoch = tk.StringVar(
            value="10"
        )
        ttk.Entry(
            parameter_frame,
            textvariable=self.epoch,
            width=15
        ).grid(
            row=1,
            column=1,
            sticky=tk.W,
            padx=5,
            pady=5
        )
        # training button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(
            fill="x",
            pady=10
        )
        self.train_button = ttk.Button(
            button_frame,
            text="开始训练",
            width=20,
            command=self.start_training
        )
        self.train_button.pack()
        # training log
        log_frame = ttk.LabelFrame(
            main_frame,
            text="训练日志",
            padding=10
        )
        log_frame.pack(
            fill="both",
            expand=True,
            pady=(0, 10)
        )
        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            font=("Consolas", 10)
        )
        self.log_text.pack(
            side="left",
            fill="both",
            expand=True
        )
        log_scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview
        )
        log_scrollbar.pack(
            side="right",
            fill="y"
        )
        self.log_text.configure(
            yscrollcommand=log_scrollbar.set
        )
        # status
        self.status_label = ttk.Label(
            main_frame,
            text="状态：等待训练"
        )
        self.status_label.pack(
            anchor="w"
        )

    def select_model(self):
        path = filedialog.askdirectory(title="选择基准模型目录")
        if path:
            self.model_path.set(path)

    def select_dataset(self):
        path = filedialog.askopenfilename(
            title="选择训练数据",
            filetypes=[
                ("JSONL 文件", "*.jsonl"),
                ("JSON 文件", "*.json"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.dataset_path.set(path)

    def select_output(self):
        path = filedialog.askdirectory(title="选择训练输出目录")
        if path:
            self.output_path.set(path)

    def start_training(self):
        model_path = self.model_path.get().strip()
        dataset_path = self.dataset_path.get().strip()
        output_path = self.output_path.get().strip()
        device = self.device.get()
        # ===============================
        # 参数检查
        # ===============================
        if not model_path:
            messagebox.showerror(
                "错误",
                "请选择基准模型。"
            )
            return
        if not dataset_path:
            messagebox.showerror(
                "错误",
                "请选择训练数据。"
            )
            return
        if not output_path:
            messagebox.showerror(
                "错误",
                "请选择输出目录。"
            )
            return
        try:
            epochs = int(self.epoch.get())
            if epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "错误",
                "训练轮数必须是大于 0 的整数。"
            )
            return
        # ===============================
        # 清空日志
        # ===============================
        self.log_text.delete(
            "1.0",
            tk.END
        )
        self.status_label.config(
            text="状态：训练中..."
        )
        self.train_button.config(
            state="disabled"
        )
        # ===============================
        # 创建训练器
        # ===============================
        trainer = TrainerEngine(
            model_path=model_path,
            dataset_path=dataset_path,
            output_path=output_path,
            device=device,
            epochs=epochs,
            log_callback=self.log_queue.put
        )
        # ===============================
        # 后台线程
        # ===============================
        training_thread = threading.Thread(
            target=self.run_training,
            args=(trainer,),
            daemon=True
        )
        training_thread.start()

    def run_training(self, trainer):
        try:
            trainer.train()
            self.log_queue.put(
                ("SUCCESS", "训练完成")
            )
        except Exception as e:
            self.log_queue.put(
                ("ERROR", str(e))
            )

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                # ===============================
                # 普通日志
                # ===============================
                if isinstance(message, str):
                    self.log_text.insert(
                        tk.END,
                        message + "\n"
                    )
                    self.log_text.see(
                        tk.END
                    )
                # ===============================
                # 特殊消息
                # ===============================
                elif isinstance(message, tuple):
                    message_type, content = message
                    if message_type == "SUCCESS":
                        self.log_text.insert(
                            tk.END,
                            "\n训练成功。\n"
                        )
                        self.status_label.config(
                            text="状态：训练完成"
                        )
                        self.train_button.config(
                            state="normal"
                        )
                        messagebox.showinfo(
                            "训练完成",
                            "训练已经完成。\n\n"
                            f"模型已经保存到：\n"
                            f"{self.output_path.get()}"
                        )
                    elif message_type == "ERROR":
                        self.log_text.insert(
                            tk.END,
                            "\n训练失败：\n"
                            + content
                            + "\n"
                        )
                        self.status_label.config(
                            text="状态：训练失败"
                        )
                        self.train_button.config(
                            state="normal"
                        )
                        messagebox.showerror(
                            "训练失败",
                            content
                        )
                    self.log_text.see(
                        tk.END
                    )
        except queue.Empty:
            pass
        self.root.after(
            100,
            self.process_log_queue
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainUI(root)
    root.mainloop()
