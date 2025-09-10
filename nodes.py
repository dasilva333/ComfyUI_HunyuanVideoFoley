import os
import torch
import torch.nn as nn
import torchaudio
import tempfile
import numpy as np
from loguru import logger
from typing import Optional, Tuple
import folder_paths
import random
from datetime import datetime
import time
import comfy.model_management as mm
import comfy.utils

# Merged Imports: Includes his model_management and your forked utils
try:
    from hunyuanvideo_foley.utils.model_utils import load_model as original_load_model
    from .utils import denoise_process_safely, feature_process_unified, extract_video_path, create_node_exit_values, load_model
    from hunyuanvideo_foley.utils.media_utils import merge_audio_video
    from .model_management import find_or_download, get_model_dir
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    raise

# Add foley models directory to ComfyUI folder paths
foley_models_dir = os.path.join(folder_paths.models_dir, "foley")
if "foley" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["foley"] = ([foley_models_dir], folder_paths.supported_pt_extensions)

class HunyuanVideoFoleyNode:
    _model_dict = None; _cfg = None; _device = None
    _model_path = None; _memory_efficient = False
    
    @classmethod
    def INPUT_TYPES(cls):
        # Using his more feature-rich INPUT_TYPES
        return {
            "required": {
                "text_prompt": ("STRING", {"multiline": True, "default": "footstep sound, impact, water splash"}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 200, "step": 1}),
                "sample_nums": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "video": ("VIDEO",), "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "output_format": (["video_path", "frames", "both"], {"default": "both"}),
                "output_folder": ("STRING", {"default": "hunyuan_foley"}),
                "filename_prefix": ("STRING", {"default": "foley_"}),
                "feature_extraction_batch_size": ("INT", {"default": 0, "min": 0, "max": 128, "step": 2, "tooltip": "Frames to process at once. 0 for auto."}),
                "syncformer_batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1, "tooltip": "Internal batch size for Syncformer."}),
                "enable_profiling": ("BOOLEAN", {"default": False}),
                "memory_efficient": ("BOOLEAN", {"default": False}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "enabled": ("BOOLEAN", {"default": True}),
                "silent_audio": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("video_path", "video_frames", "audio", "status_message")
    FUNCTION = "generate_audio"
    CATEGORY = "HunyuanVideo-Foley"
    
    @classmethod
    def setup_device(cls):
        return mm.get_torch_device()

    @classmethod
    def load_models(cls, memory_efficient: bool = False, cpu_offload: bool = False) -> Tuple[bool, str]:
        try:
            # His robust model management and auto-downloading
            logger.info("Verifying local model integrity...")
            find_or_download("hunyuanvideo_foley.pth", "Tencent-Hunyuan/HunyuanVideo-Foley", "hunyuanvideo-foley-xxl")
            find_or_download("vae_128d_48k.pth", "Tencent-Hunyuan/HunyuanVideo-Foley", "hunyuanvideo-foley-xxl")
            find_or_download("synchformer_state_dict.pth", "Tencent-Hunyuan/HunyuanVideo-Foley", "hunyuanvideo-foley-xxl")
            from .model_management import get_siglip_path, get_clap_path
            get_siglip_path(); get_clap_path() # Ensures Hugging Face cache is populated
            
            model_dir = get_model_dir("hunyuanvideo-foley-xxl")
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "hunyuanvideo-foley-xxl.yaml")

            if (cls._model_dict and cls._cfg and (cls._model_path == model_dir or cls._model_path == "preloaded") and not memory_efficient):
                return True, "Models already loaded"
            
            # Your robust, memory-safe loading logic
            cls._device = torch.device("cpu") # Always load to CPU first
            logger.info(f"Loading models from directory: {model_dir} to CPU")
            
            # This now calls our memory-safe `load_model` from utils.py
            cls._model_dict, cls._cfg = load_model(model_dir, config_path, cls._device)
            
            if cls._model_path != "preloaded": cls._model_path = model_dir
            cls._memory_efficient = memory_efficient
            
            return True, "Models loaded successfully!"
        except Exception as e:
            import traceback
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg); logger.error(traceback.format_exc())
            cls._model_dict, cls._cfg, cls._device, cls._model_path = None, None, None, None
            return False, error_msg
    
    def set_seed(self, seed: int):
        seed = int(seed) % (2**32)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    @classmethod
    def _extract_frames_from_image_input(cls, images, fps=24.0):
        # Merged version with his data type fix
        import cv2
        try:
            if images is None: return None, "No images provided"
            frames = images.cpu().numpy()
            if frames.dtype != np.uint8:
                frames = (frames.clip(0, 1) * 255).astype(np.uint8)
            if frames.shape[-1] == 4:
                frames = frames[..., :3]
            batch_size, height, width, _ = frames.shape
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4'); os.close(temp_fd)
            out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not out.isOpened(): return None, "Failed to open video writer"
            for i in range(batch_size): out.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            out.release()
            return temp_path, "Success"
        except Exception as e:
            return None, f"Error converting images to video: {str(e)}"
    
    @torch.inference_mode()
    def generate_audio(self, **kwargs):
        # Your robust, unified `generate_audio` function using kwargs
        enabled = kwargs.get("enabled", True)
        silent_audio = kwargs.get("silent_audio", True)
        video = kwargs.get("video")
        images = kwargs.get("images")
        
        if not enabled:
            return create_node_exit_values(silent_audio, video, images, "✅ Node disabled. Skipped.")
        
        try:
            self.set_seed(kwargs.get("seed", 42))
            logger.info("Performing pre-run VRAM cleanup...")
            mm.unload_all_models(); import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            if self._model_dict is None or self._cfg is None:
                success, message = self.load_models(
                    memory_efficient=kwargs.get("memory_efficient", False), 
                    cpu_offload=kwargs.get("cpu_offload", False)
                )
                if not success: raise Exception(f"Model loading failed: {message}")
            
            if self._model_dict is None or self._cfg is None:
                raise Exception("Models not loaded, and fallback loading failed.")
            if video is None and images is None:
                raise Exception("Please provide either a video or images input!")

            
            logger.info("Processing media features...")
            visual_feats, text_feats, audio_len_in_s = feature_process_unified(
                video_input=video, 
                image_input=images, 
                model_dict=self._model_dict, 
                cfg=self._cfg, 
                prompt=kwargs.get("text_prompt"),
                negative_prompt=kwargs.get("negative_prompt", ""), 
                fps_hint=kwargs.get("fps", 24.0),
                batch_size=kwargs.get("feature_extraction_batch_size", 8),
                sync_batch_size=kwargs.get("syncformer_batch_size", 8),
                enable_profiling=kwargs.get("enable_profiling", False)
            )
            
            target_device = mm.get_torch_device()
            logger.info(f"Preparing for denoising on device: {target_device}")
            visual_feats['siglip2_feat'] = visual_feats['siglip2_feat'].to("cpu")
            visual_feats['syncformer_feat'] = visual_feats['syncformer_feat'].to("cpu")
            text_feats['text_feat'] = text_feats['text_feat'].to("cpu")
            text_feats['uncond_text_feat'] = text_feats['uncond_text_feat'].to("cpu")
            # self._model_dict.foley_model.to(target_device)
            # self._model_dict.dac_model.to(target_device)

            logger.info("Generating audio...")
            audio, sample_rate = denoise_process_safely(visual_feats, text_feats, audio_len_in_s, self._model_dict, self._cfg, **kwargs)
            
            output_dir = os.path.join(folder_paths.get_output_directory(), kwargs.get("output_folder", "hunyuan_foley"))
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"{kwargs.get('filename_prefix', 'foley_')}audio_{timestamp}_{kwargs.get('seed', 42)}.wav"
            audio_output_path = os.path.join(output_dir, audio_filename)
            torchaudio.save(audio_output_path, audio[0], sample_rate)
            audio_result = {"waveform": audio[0].unsqueeze(0), "sample_rate": sample_rate}
            
            video_output_path, video_frames_out = ("", torch.zeros((1, 1, 1, 3), dtype=torch.float32))
            temp_video_for_output, source_video_path = (None, extract_video_path(video))
            if not source_video_path and images is not None:
                temp_video_for_output, _ = self._extract_frames_from_image_input(images, kwargs.get("fps", 24.0))
                source_video_path = temp_video_for_output
            
            if kwargs.get("output_format") in ["video_path", "both"] and source_video_path:
                video_filename = f"{kwargs.get('filename_prefix', 'foley_')}video_{timestamp}_{kwargs.get('seed', 42)}.mp4"
                video_output_path = os.path.join(output_dir, video_filename)
                try: merge_audio_video(audio_output_path, source_video_path, video_output_path)
                except Exception as e: logger.error(f"Failed to merge audio and video: {e}"); video_output_path = source_video_path
            
            if kwargs.get("output_format") in ["frames", "both"]:
                if images is not None: video_frames_out = images
                elif source_video_path:
                    try:
                        import cv2
                        cap = cv2.VideoCapture(source_video_path); frames_list = []
                        while cap.isOpened():
                            ret, frame = cap.read();
                            if not ret: break
                            frames_list.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.float32) / 255.0)
                        cap.release()
                        if frames_list: video_frames_out = torch.from_numpy(np.stack(frames_list))
                    except Exception as e: logger.warning(f"Could not extract frames for output: {e}")

            if temp_video_for_output and os.path.exists(temp_video_for_output):
                try: os.remove(temp_video_for_output)
                except: pass

            return (video_output_path, video_frames_out, audio_result, "✅ Generated audio successfully")
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg); logger.error(traceback.format_exc())
            return create_node_exit_values(silent_audio, video, images, error_msg)
        
        finally:
            # Your robust "Good Neighbor" cleanup logic
            logger.info("HunyuanVideo-Foley: Starting guaranteed cleanup...")
            if kwargs.get("memory_efficient", False) or kwargs.get("cpu_offload", False):
                if self._model_dict:
                    logger.info("Offloading models to CPU...")
                    for model_obj in self._model_dict.values():
                        if hasattr(model_obj, 'to'):
                            try: model_obj.to("cpu")
                            except: pass
            
            if kwargs.get("memory_efficient", False) and self._model_path != "preloaded":
                logger.info("Memory efficient mode: Unloading models completely.")
                self._model_dict, self._cfg, self._model_path = None, None, None
            
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info("HunyuanVideo-Foley: Cleanup complete.")
        
class LinearFP8Wrapper(nn.Module):
    # This class is unchanged from upstream
    def __init__(self, original_linear, dtype="fp8_e4m3fn"):
        super().__init__()
        self.dtype, self.bias = dtype, original_linear.bias
        if dtype == "fp8_e4m3fn": self.weight_fp8 = original_linear.weight.to(torch.float8_e4m3fn)
        elif dtype == "fp8_e5m2": self.weight_fp8 = original_linear.weight.to(torch.float8_e5m2)
        else: self.weight_fp8 = original_linear.weight
    def forward(self, x): return F.linear(x, self.weight_fp8.to(x.dtype), self.bias)

class HunyuanVideoFoleyModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "quantization": (["none", "fp8_e4m3fn", "fp8_e5m2"], {"default": "none"}),
            "cpu_offload": ("BOOLEAN", {"default": False}),
        }, "optional": {"model_path": ("STRING", {"default": ""}), "config_path": ("STRING", {"default": ""})}}
    
    RETURN_TYPES = ("FOLEY_MODEL", "STRING")
    FUNCTION = "load_model"
    CATEGORY = "HunyuanVideo-Foley/Loaders"
    
    def load_model(self, quantization="none", cpu_offload=False, model_path="", config_path=""):
        try:
            # This now cleanly calls the main node's loader
            success, message = HunyuanVideoFoleyNode.load_models(model_path, config_path, cpu_offload=cpu_offload)
            if not success: raise Exception(message)
            
            model_dict = HunyuanVideoFoleyNode._model_dict
            if quantization != "none" and hasattr(model_dict, "dac_model"):
                logger.info(f"Applying {quantization} quantization to VAE...")
                vae_model = model_dict.dac_model
                for name, module in vae_model.named_modules():
                    if isinstance(module, nn.Linear):
                        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                        child_name = name.split('.')[-1]
                        parent = vae_model.get_submodule(parent_name) if parent_name else vae_model
                        setattr(parent, child_name, LinearFP8Wrapper(module, quantization))
            
            model_info = {
                "model_dict": model_dict, "cfg": HunyuanVideoFoleyNode._cfg, 
                "device": HunyuanVideoFoleyNode._device, "quantization": quantization, 
                "cpu_offload": cpu_offload
            }
            return (model_info, f"✅ Model loaded with {quantization} quantization")
        except Exception as e:
            return (None, f"❌ Failed to load model: {str(e)}")

class HunyuanVideoFoleyDependenciesLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("FOLEY_MODEL",)},}
    
    RETURN_TYPES = ("FOLEY_DEPS", "STRING")
    FUNCTION = "load_dependencies"
    CATEGORY = "HunyuanVideo-Foley/Loaders"
    
    def load_dependencies(self, model):
        if model and isinstance(model, dict) and "model_dict" in model:
            return (model, "✅ Dependencies ready")
        return (None, "❌ Invalid model input")

class HunyuanVideoFoleyTorchCompile:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dependencies": ("FOLEY_DEPS",), "compile_vae": ("BOOLEAN", {"default": True}),
            "compile_mode": (["default", "reduce-overhead", "max-autotune"], {"default": "default"}),
            "backend": (["inductor", "cudagraphs", "eager"], {"default": "inductor"}),
        }}
    
    RETURN_TYPES = ("FOLEY_COMPILED", "STRING")
    FUNCTION = "compile_model"
    CATEGORY = "HunyuanVideo-Foley/Optimization"
    
    def compile_model(self, dependencies, compile_vae=True, compile_mode="default", backend="inductor"):
        try:
            if not (dependencies and isinstance(dependencies, dict) and "model_dict" in dependencies):
                return (None, "❌ Invalid dependencies input")
            
            compiled_deps = dependencies.copy()
            compiled_deps["model_dict"] = dependencies["model_dict"].copy()
            model_dict = compiled_deps["model_dict"]

            if compile_vae and backend != "eager" and hasattr(model_dict, 'dac_model'):
                logger.info(f"Compiling DAC VAE with mode={compile_mode}, backend={backend}...")
                import torch._dynamo as dynamo; dynamo.config.suppress_errors = True
                model_dict.dac_model = torch.compile(model_dict.dac_model, mode=compile_mode, backend=backend)
                status = f"✅ DAC VAE compiled"
            else:
                status = "✅ Model ready (compilation skipped)"
            
            return (compiled_deps, status)
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            return (dependencies, f"⚠️ Compilation failed, using uncompiled: {str(e)}")

class HunyuanVideoFoleyGeneratorAdvanced(HunyuanVideoFoleyNode):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["optional"]["compiled_model"] = ("FOLEY_COMPILED",)
        return base_inputs
    
    FUNCTION = "generate_audio_advanced"
    CATEGORY = "HunyuanVideo-Foley"
    
    def generate_audio_advanced(self, **kwargs):
        compiled_model = kwargs.pop("compiled_model", None)
        
        if compiled_model and isinstance(compiled_model, dict):
            if isinstance(compiled_model, tuple): compiled_model = compiled_model[0]
            if compiled_model and "model_dict" in compiled_model:
                self.__class__._model_dict = compiled_model.get("model_dict")
                self.__class__._cfg = compiled_model.get("cfg")
                self.__class__._device = compiled_model.get("device")
                self.__class__._model_path = "preloaded"
                self.__class__._memory_efficient = kwargs.get("memory_efficient", False)
        else:
            # If no model is piped in, or it was invalid, clear state to force fresh load.
            if self.__class__._model_path == "preloaded" or not compiled_model:
                 self.__class__._model_dict, self.__class__._cfg, self.__class__._model_path = None, None, None
        
        return self.generate_audio(**kwargs)

NODE_CLASS_MAPPINGS = {
    "HunyuanVideoFoley": HunyuanVideoFoleyNode,
    "HunyuanVideoFoleyModelLoader": HunyuanVideoFoleyModelLoader,
    "HunyuanVideoFoleyDependenciesLoader": HunyuanVideoFoleyDependenciesLoader,
    "HunyuanVideoFoleyTorchCompile": HunyuanVideoFoleyTorchCompile,
    "HunyuanVideoFoleyGeneratorAdvanced": HunyuanVideoFoleyGeneratorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoFoley": "HunyuanVideo-Foley Generator",
    "HunyuanVideoFoleyModelLoader": "HunyuanVideo-Foley Model Loader",
    "HunyuanVideoFoleyDependenciesLoader": "HunyuanVideo-Foley Dependencies",
    "HunyuanVideoFoleyTorchCompile": "HunyuanVideo-Foley Torch Compile",
    "HunyuanVideoFoleyGeneratorAdvanced": "HunyuanVideo-Foley Generator (Advanced)",
}