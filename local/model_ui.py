import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_engine import ModelEngine


# =========================================================
# API
# =========================================================

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7


app = FastAPI()

engine = ModelEngine()


@app.get("/status")
def status():
    return {
        "loaded": engine.model is not None
    }

@app.post("/chat")
def chat(request: ChatRequest):
    if engine.model is None:
        raise HTTPException(
            status_code=503,
            detail="模型尚未加载"
        )
    try:
        result = engine.chat(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        return {
            "success": True,
            "response": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================================================
# UI
# =========================================================

class ModelUI:

    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        engine.log_callback = self.log_queue.put
        self.root.after(
            100,
            self.process_log_queue
        )
        self.root.title("AI模型加载工具")
        self.root.geometry("1024x768")
        self.root.minsize(900, 650)
        main_frame = ttk.Frame(
            self.root,
            padding=15
        )
        main_frame.pack(
            fill="both",
            expand=True
        )
        # =================================================
        # Title
        # =================================================
        title_label = ttk.Label(
            main_frame,
            text="AI模型加载工具",
            font=("Microsoft YaHei", 18, "bold"),
            anchor="center"
        )
        title_label.pack(
            fill="x",
            pady=(0, 15)
        )
        # =================================================
        # Base Model
        # =================================================
        model_frame = ttk.LabelFrame(
            main_frame,
            text="基础 AI 模型",
            padding=10
        )
        model_frame.pack(
            fill="x",
            pady=(0, 10)
        )
        ttk.Label(
            model_frame,
            text="模型路径："
        ).grid(
            row=0,
            column=0,
            padx=(0, 8),
            pady=5
        )
        self.model_path = tk.StringVar()
        ttk.Entry(
            model_frame,
            textvariable=self.model_path
        ).grid(
            row=0,
            column=1,
            sticky=tk.EW,
            pady=5
        )
        ttk.Button(
            model_frame,
            text="选择...",
            width=10,
            command=self.select_model
        ).grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )
        model_frame.columnconfigure(
            1,
            weight=1
        )
        # =================================================
        # LoRA
        # =================================================
        lora_frame = ttk.LabelFrame(
            main_frame,
            text="增量训练模型",
            padding=10
        )
        lora_frame.pack(
            fill="x",
            pady=(0, 10)
        )
        ttk.Label(
            lora_frame,
            text="LoRA 路径："
        ).grid(
            row=0,
            column=0,
            padx=(0, 8),
            pady=5
        )
        self.lora_path = tk.StringVar()
        ttk.Entry(
            lora_frame,
            textvariable=self.lora_path
        ).grid(
            row=0,
            column=1,
            sticky=tk.EW,
            pady=5
        )
        ttk.Button(
            lora_frame,
            text="选择...",
            width=10,
            command=self.select_lora
        ).grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )
        lora_frame.columnconfigure(
            1,
            weight=1
        )
        # =================================================
        # Load Button
        # =================================================
        button_frame = ttk.Frame(
            main_frame
        )
        button_frame.pack(
            fill="x",
            pady=10
        )
        self.load_button = ttk.Button(
            button_frame,
            text="加载模型",
            width=20,
            command=self.start_loading
        )
        self.load_button.pack()
        # =================================================
        # Log
        # =================================================
        log_frame = ttk.LabelFrame(
            main_frame,
            text="模型日志",
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
        scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview
        )
        scrollbar.pack(
            side="right",
            fill="y"
        )
        self.log_text.configure(
            yscrollcommand=scrollbar.set
        )
        # =================================================
        # Status
        # =================================================
        self.status_label = ttk.Label(
            main_frame,
            text="状态：等待加载模型"
        )
        self.status_label.pack(
            anchor="w"
        )

    # =====================================================
    # Select Base Model
    # =====================================================

    def select_model(self):
        path = filedialog.askdirectory(
            title="选择基础 AI 模型目录"
        )
        if path:
            self.model_path.set(path)

    # =====================================================
    # Select LoRA
    # =====================================================

    def select_lora(self):
        path = filedialog.askdirectory(
            title="选择 LoRA 增量模型目录"
        )
        if path:
            self.lora_path.set(path)

    # =====================================================
    # Start Loading
    # =====================================================

    def start_loading(self):
        model_path = self.model_path.get().strip()
        lora_path = self.lora_path.get().strip()
        if not model_path:
            messagebox.showerror(
                "错误",
                "请选择基础 AI 模型。"
            )
            return
        if not lora_path:
            messagebox.showerror(
                "错误",
                "请选择 LoRA 增量模型。"
            )
            return
        self.log_text.delete(
            "1.0",
            tk.END
        )
        self.status_label.config(
            text="状态：正在加载模型..."
        )
        self.load_button.config(
            state="disabled"
        )
        thread = threading.Thread(
            target=self.load_model,
            args=(model_path, lora_path),
            daemon=True
        )
        thread.start()

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(
            self,
            model_path,
            lora_path
    ):
        try:
            engine.load_model(
                base_model_path=model_path,
                lora_path=lora_path
            )
            self.log_queue.put((
                "SUCCESS",
                "模型加载完成"
            )
            )
        except Exception as e:
            self.log_queue.put(
                (
                    "ERROR",
                    str(e)
                )
            )

    # =====================================================
    # Process Log
    # =====================================================

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if isinstance(
                        message,
                        str
                ):
                    self.log_text.insert(
                        tk.END,
                        message + "\n"
                    )
                    self.log_text.see(
                        tk.END
                    )
                elif isinstance(
                        message,
                        tuple
                ):
                    message_type, content = message
                    if message_type == "SUCCESS":
                        self.log_text.insert(
                            tk.END,
                            "\n模型加载成功。\n"
                        )
                        self.status_label.config(
                            text="状态：模型已加载，API 服务运行中"
                        )
                        self.load_button.config(
                            state="normal"
                        )
                        messagebox.showinfo(
                            "完成",
                            "基础模型和 LoRA 增量模型加载完成。可以通过访问/chat接口和本地AI进行对话"
                        )
                    elif message_type == "ERROR":
                        self.log_text.insert(
                            tk.END,
                            "\n模型加载失败：\n"
                            + content
                            + "\n"
                        )
                        self.status_label.config(
                            text="状态：模型加载失败"
                        )
                        self.load_button.config(
                            state="normal"
                        )
                        messagebox.showerror(
                            "模型加载失败",
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


# =========================================================
# Start API Server
# =========================================================

def start_api_server():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )


# =========================================================
# Start
# =========================================================

if __name__ == "__main__":
    api_thread = threading.Thread(
        target=start_api_server,
        daemon=True
    )
    api_thread.start()
    root = tk.Tk()
    app_ui = ModelUI(root)
    root.mainloop()
