import argparse
import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from inference.encoder_head_spans_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_METRICS_PATH,
    load_metrics,
    make_head_from_checkpoint,
    resolve_runtime_config,
    score_records,
)
from source.utils.LLM_train import (  # noqa: E402
    load_backbone,
    prepare_tokenizer,
    require_torch_and_encoder_transformers,
    require_torch_and_transformers,
    select_device,
    select_dtype,
)
from source.utils.text import QUERY_FIELD, RESPONSE_FIELD  # noqa: E402


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MirageIndex local chat demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind. Defaults to 7860.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL, help="Qwen model id.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    return parser.parse_args()


class ModelService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.loaded = False
        self.torch = None
        self.encoder_parts: dict[str, Any] = {}
        self.qwen_parts: dict[str, Any] = {}

    def load(self) -> None:
        with self.lock:
            if self.loaded:
                return

            torch, nn, DataLoader, Dataset, AutoModel, EncoderTokenizer = require_torch_and_encoder_transformers(
                "demo_encoder_inference"
            )
            _torch, _nn, _DataLoader, _Dataset, AutoModelForCausalLM, QwenTokenizer = require_torch_and_transformers(
                "demo_qwen_generation"
            )
            device = select_device(self.args.device, torch)
            dtype = select_dtype(self.args.torch_dtype, device, torch)

            metrics = load_metrics(self.args.metrics)
            checkpoint = torch.load(self.args.checkpoint, map_location=device)
            config_args = SimpleNamespace(aggregation=None, top_k=None, threshold=None, batch_size=None)
            config = resolve_runtime_config(config_args, checkpoint, metrics)
            config["include_chunks"] = False

            encoder_tokenizer = prepare_tokenizer(config["base_model"], EncoderTokenizer)
            encoder_backbone = load_backbone(config["base_model"], dtype, device, AutoModel)
            encoder_backbone.eval()
            for parameter in encoder_backbone.parameters():
                parameter.requires_grad = False
            encoder_head = make_head_from_checkpoint(checkpoint, nn, device)

            qwen_tokenizer = QwenTokenizer.from_pretrained(self.args.qwen_model)
            if qwen_tokenizer.pad_token is None:
                qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
            qwen_model = AutoModelForCausalLM.from_pretrained(self.args.qwen_model, torch_dtype=dtype)
            qwen_model.to(device)
            qwen_model.eval()
            if hasattr(qwen_model.config, "use_cache"):
                qwen_model.config.use_cache = True

            self.torch = torch
            self.encoder_parts = {
                "tokenizer": encoder_tokenizer,
                "backbone": encoder_backbone,
                "head": encoder_head,
                "config": config,
                "DataLoader": DataLoader,
                "Dataset": Dataset,
                "device": device,
            }
            self.qwen_parts = {
                "tokenizer": qwen_tokenizer,
                "model": qwen_model,
                "device": device,
            }
            self.loaded = True

    def answer(self, question: str) -> str:
        self.load()
        tokenizer = self.qwen_parts["tokenizer"]
        model = self.qwen_parts["model"]
        device = self.qwen_parts["device"]
        torch = self.torch

        messages = [{"role": "user", "content": question}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            inputs = {"input_ids": input_ids.to(device)}
        except TypeError:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": input_ids.to(device)}
        except Exception:
            inputs = tokenizer(question, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

        generation_args = {
            **inputs,
            "max_new_tokens": self.args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if self.args.temperature > 0:
            generation_args.update({"do_sample": True, "temperature": self.args.temperature, "top_p": 0.9})
        else:
            generation_args.update({"do_sample": False})

        with torch.no_grad():
            output_ids = model.generate(**generation_args)

        prompt_length = generation_args["input_ids"].shape[-1]
        answer_ids = output_ids[0][prompt_length:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return answer or "[empty response]"

    def score(self, question: str, answer: str) -> dict[str, Any]:
        self.load()
        torch = self.torch
        parts = self.encoder_parts
        records = [{QUERY_FIELD: question, RESPONSE_FIELD: answer}]
        result = score_records(
            records=records,
            tokenizer=parts["tokenizer"],
            backbone=parts["backbone"],
            head=parts["head"],
            config=parts["config"],
            torch=torch,
            DataLoader=parts["DataLoader"],
            Dataset=parts["Dataset"],
            device=parts["device"],
        )[0]
        return result

    def chat(self, question: str) -> dict[str, Any]:
        answer = self.answer(question)
        score = self.score(question, answer)
        return {
            "answer": answer,
            "hallucination_score": score["hallucination_score"],
            "prediction": score["prediction"],
            "threshold": score["threshold"],
            "aggregation": score["aggregation"],
            "top_k": score["top_k"],
            "chunk_count": score["chunk_count"],
        }


def make_handler(service: ModelService) -> type[BaseHTTPRequestHandler]:
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                self.send_file(DEMO_ROOT / "index.html", "text/html; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                payload = self.read_json()
                question = str(payload.get("question", "")).strip()
                if not question:
                    raise ValueError("question is required")
                result = service.chat(question)
                self.send_json(result)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            return payload

        def send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def main() -> None:
    args = parse_args()
    if args.port <= 0:
        raise ValueError("--port must be greater than 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be greater than or equal to 0")

    service = ModelService(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    url = f"http://{args.host}:{args.port}/"
    print(f"MirageIndex demo running at {url}")
    print("The first message will load Qwen3 and ModernBERT, so it can take a while.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
