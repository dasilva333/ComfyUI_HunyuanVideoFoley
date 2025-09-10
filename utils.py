"""
Utility functions for ComfyUI HunyuanVideo-Foley custom node
"""

import os
import tempfile
import torch
import numpy as np
from typing import Union, Optional, Tuple
from loguru import logger
import decord
from tqdm import tqdm
from PIL import Image
from einops import rearrange
import comfy.model_management as mm
# We need to import the original library functions that our safe wrappers will call.
from hunyuanvideo_foley.utils.feature_utils import encode_text_feat
from hunyuanvideo_foley.utils.config_utils import AttributeDict
import time

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torchvision import transforms
from torchvision.transforms import v2
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection
from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
from hunyuanvideo_foley.models.synchformer import Synchformer
from hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
from hunyuanvideo_foley.utils.config_utils import load_yaml
from hunyuanvideo_foley.utils.model_utils import load_state_dict

# In utils.py, add this function

def load_model(model_path, config_path, device):
    """
    Memory-safe model loader forked from the original library.
    Uses accelerate for the large Foley model to prevent CPU RAM OOM errors.
    """
    logger.info("Starting memory-safe model loading process...")
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Model weights dir: {model_path}")
    logger.info(f"Target device: {device}")
    
    cfg = load_yaml(config_path)
    logger.info("Configuration loaded successfully")
    
    # --- 1. HunyuanVideoFoley (Loaded with Accelerate) ---
    logger.info("Loading HunyuanVideoFoley main model with accelerate...")
    with init_empty_weights():
        # Create the model structure on the 'meta' device without allocating memory
        foley_model = HunyuanVideoFoley(cfg, dtype=torch.bfloat16, device="cpu")

    foley_checkpoint_path = os.path.join(model_path, "hunyuanvideo_foley.pth")
    # Intelligently load weights from disk directly into the model's layers,
    # loading to CPU to keep VRAM free for now.
    # THIS IS THE CRITICAL FIX for VRAM issues.
    # "auto" lets accelerate split the model between GPU and CPU to fit into memory.
    load_checkpoint_and_dispatch(
        foley_model, 
        checkpoint=foley_checkpoint_path, 
        device_map="auto",
        no_split_module_classes=foley_model._no_split_modules
    )
    foley_model.eval()
    logger.info("HunyuanVideoFoley model loaded successfully to CPU.")

    # --- 2. DAC-VAE (Loaded normally to CPU) ---
    dac_path = os.path.join(model_path, "vae_128d_48k.pth")
    logger.info(f"Loading DAC VAE model from: {dac_path}")
    dac_model = DAC.load(dac_path, map_location="cpu")
    dac_model.requires_grad_(False).eval()
    logger.info("DAC VAE model loaded successfully to CPU.")

    # --- 3. Siglip2 visual-encoder (Loaded normally to CPU) ---
    logger.info("Loading SigLIP2 visual encoder...")
    siglip2_preprocess = transforms.Compose([
                transforms.Resize((512, 512)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    siglip2_model = AutoModel.from_pretrained("google/siglip2-base-patch16-512").eval().to(dtype=torch.bfloat16).to("cpu")
    logger.info("SigLIP2 model loaded successfully to CPU.")

    # --- 4. CLAP text-encoder (Loaded normally to CPU) ---
    logger.info("Loading CLAP text encoder...")
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
    clap_model = ClapTextModelWithProjection.from_pretrained("laion/larger_clap_general").to("cpu")
    logger.info("CLAP tokenizer and model loaded successfully to CPU.")

    # --- 5. Syncformer (Loaded normally to CPU) ---
    syncformer_path = os.path.join(model_path, "synchformer_state_dict.pth")
    # In the `load_model` function, replace the syncformer loading block

    # 5. syncformer
    logger.info(f"Loading Synchformer model (CPU -> GPU)...")
    syncformer_path = os.path.join(model_path, "synchformer_state_dict.pth")
    syncformer_preprocess = v2.Compose(
        [ v2.Resize(224), v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
          v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ]
    )
    # This is the fix: Initialize, load weights on CPU, then cast the entire loaded model to bfloat16.
    syncformer_model = Synchformer().to("cpu")
    syncformer_model.load_state_dict(torch.load(syncformer_path, map_location="cpu"))
    # This is the fix: Load the model in float16 to match the environment's input type.
    syncformer_model.to(dtype=torch.float16).eval()
    logger.info("Synchformer model loaded successfully to CPU.")

    model_dict = AttributeDict({
        'foley_model': foley_model, 'dac_model': dac_model,
        'siglip2_preprocess': siglip2_preprocess, 'siglip2_model': siglip2_model,
        'clap_tokenizer': clap_tokenizer, 'clap_model': clap_model,
        'syncformer_preprocess': syncformer_preprocess, 'syncformer_model': syncformer_model,
        'device': device,
    })
    
    logger.info("All models loaded successfully to CPU and packaged in model_dict.")
    return model_dict, cfg

class SimpleProfiler:
    """A simple profiler for timing and CUDA memory logging."""
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        self.start_time = None
        self.device = get_optimal_device()

    def __enter__(self):
        if not self.enabled: return self
        self.start_time = time.time()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            start_mem = torch.cuda.memory_allocated(self.device) / 1024**2
            logger.info(f"[Profiler:{self.name}] Start VRAM: {start_mem:.2f} MB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled: return
        elapsed = time.time() - self.start_time
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            end_mem = torch.cuda.memory_allocated(self.device) / 1024**2
            logger.info(f"[Profiler:{self.name}] Time: {elapsed:.4f}s. End VRAM: {end_mem:.2f} MB")
        else:
            logger.info(f"[Profiler:{self.name}] Time: {elapsed:.4f}s. (CPU)")

def tensor_to_video(video_tensor: torch.Tensor, output_path: str, fps: int = 30) -> str:
    """
    Convert a video tensor to a video file
    """
    try:
        import cv2
        video_np = video_tensor.detach().cpu().numpy() if isinstance(video_tensor, torch.Tensor) else np.array(video_tensor)
        
        if video_np.ndim == 4 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
        elif video_np.ndim == 5 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np[0], (0, 2, 3, 1))
        
        video_np = (video_np * 255).astype(np.uint8) if video_np.max() <= 1.0 else video_np.astype(np.uint8)
        
        frames, height, width, channels = video_np.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(frames):
            frame = video_np[i]
            if channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert tensor to video: {e}")
        raise

def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}

def ensure_video_file(video_input: Union[str, torch.Tensor, np.ndarray]) -> str:
    """
    Ensure the video input is converted to a file path
    """
    if isinstance(video_input, str):
        if os.path.exists(video_input): return video_input
        else: raise FileNotFoundError(f"Video file not found: {video_input}")
    elif isinstance(video_input, (torch.Tensor, np.ndarray)):
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "input_video.mp4")
        return tensor_to_video(video_input, output_path)
    else:
        raise ValueError(f"Unsupported video input type: {type(video_input)}")

def validate_model_files(model_path: str) -> Tuple[bool, str]:
    """
    Validate that all required model files exist
    """
    required_files = ["hunyuanvideo_foley.pth", "vae_128d_48k.pth", "synchformer_state_dict.pth"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        return False, f"Missing model files: {', '.join(missing_files)}"
    return True, "All required model files found"

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for model execution
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Using CUDA device: {props.name} with {props.total_memory / 1e9:.1f}GB memory")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return torch.device("mps")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")

def check_memory_requirements(device: torch.device, required_gb: float = 16.0) -> Tuple[bool, str]:
    """
    Check if the device has enough memory for model execution
    """
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        if total_memory < required_gb:
            return False, f"GPU has {total_memory:.1f}GB memory, but {required_gb}GB is recommended"
        else:
            return True, f"GPU has {total_memory:.1f}GB memory (sufficient)"
    return True, "Using non-CUDA device (memory check not applicable)"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    """
    seconds = float(seconds)
    if seconds < 60: return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60: return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"

# --- NEW MEMORY-SAFE AND UNIFIED FUNCTIONS ---

def extract_video_path(video):
    """Extract a file path from various potential video input types. Moved here to be a standalone utility."""
    if hasattr(video, '__class__') and 'VideoFromFile' in video.__class__.__name__:
        if hasattr(video, '_VideoFromFile__file'): return getattr(video, '_VideoFromFile__file')
        for attr in ['file', 'path', 'filename']:
            if hasattr(video, attr) and isinstance(getattr(video, attr), str): return getattr(video, attr)
    if isinstance(video, str): return video
    if isinstance(video, dict) and 'path' in video: return video['path']
    return None

# In utils.py, replace the existing function

@torch.inference_mode()
def _encode_visual_features_safely(frames_uint8, model_dict, fps_hint, batch_size=4, sync_batch_size=1, enable_profiling=False):
    """ Memory-safe visual feature extraction using multithreaded preprocessing. """
    dev = model_dict.device
    
    logger.info(f"--- Starting Safe Visual Feature Extraction ---")
    logger.info(f"Input frames: {len(frames_uint8)}, SigLIP Batch: {batch_size}, Sync Batch: {sync_batch_size}")

    mm.unload_all_models(); import gc; gc.collect()
    if dev.type == 'cuda': torch.cuda.empty_cache()
    
    visual_features = {}
    pil_list = [Image.fromarray(f).convert("RGB") for f in frames_uint8]
    
    # --- STAGE 1: SigLIP2 (with Multithreaded Preprocessing) ---
    all_siglip_feats = []
    try:
        logger.info("[SigLIP2] Moving model to device...")
        model_dict.siglip2_model.to(dev)
        
        from concurrent.futures import ThreadPoolExecutor
        # Use half the CPU cores for safety, just like before
        num_workers = max(1, os.cpu_count() // 2)
        logger.info(f"[SigLIP2] Using {num_workers} CPU threads for parallel preprocessing.")
        
        with SimpleProfiler("SigLIP2 Batching", enabled=enable_profiling):
            for i in tqdm(range(0, len(pil_list), batch_size), desc="Processing SigLIP2 batches"):
                batch_pils = pil_list[i:i+batch_size]
                
                # --- MULTITHREADED PREPROCESSING ---
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    siglip_list = list(executor.map(model_dict.siglip2_preprocess, batch_pils))
                # --- END OF MULTITHREADED ---

                clip_frames = torch.stack(siglip_list, dim=0).to(dev)
                
                batch_feat = _encode_video_with_siglip2_safely(clip_frames, model_dict)
                all_siglip_feats.append(batch_feat.cpu())
                del batch_pils, siglip_list, clip_frames, batch_feat
        
        if all_siglip_feats:
            visual_features['siglip2_feat'] = torch.cat(all_siglip_feats, dim=0).unsqueeze(0).to(dev)
            
    finally:
        logger.info("[SigLIP2] Offloading model from device."); model_dict.siglip2_model.to("cpu")
        del all_siglip_feats
        if dev.type == 'cuda': torch.cuda.empty_cache()
    # --- END STAGE 1 ---

    # --- STAGE 2: Syncformer ---
    try:
        # NOTE: The actual model-to-device move is now handled inside the library function for robustness.
        with SimpleProfiler("Syncformer Processing", enabled=enable_profiling):
            images = torch.from_numpy(np.array(frames_uint8)).permute(0, 3, 1, 2)
            sync_frames = model_dict.syncformer_preprocess(images).unsqueeze(0).to(dev, dtype=torch.float16)
            visual_features['syncformer_feat'] = encode_video_with_sync(sync_frames, model_dict, batch_size=sync_batch_size)
    finally:
        logger.info("[Syncformer] Offloading model from device."); model_dict.syncformer_model.to("cpu")
        if dev.type == 'cuda': torch.cuda.empty_cache()
    # --- END STAGE 2 ---

    audio_len_in_s = len(frames_uint8) / fps_hint
    return AttributeDict(visual_features), audio_len_in_s

def feature_process_unified(video_input, image_input, prompt, model_dict, cfg, negative_prompt="", fps_hint=24.0, batch_size=4, sync_batch_size=1, enable_profiling=False):
    """ Unified, memory-safe feature processing for either a video path or an image tensor. """
    frames_uint8 = None; fps = fps_hint
    
    if image_input is not None:
        logger.info("Processing features from IMAGE tensor input.")
        frames_uint8 = (image_input.cpu().clip(0, 1) * 255).byte().numpy() if image_input.dtype != torch.uint8 else image_input.cpu().numpy()
    elif video_input is not None:
        logger.info("Processing features from VIDEO path input.")
        video_path = extract_video_path(video_input)
        if not video_path: raise ValueError("Invalid video path provided.")
        
        video_reader = decord.VideoReader(video_path)
        total_frames = len(video_reader)
        fps = video_reader.get_avg_fps()
        
        frame_indices = np.linspace(0, total_frames - 1, num=total_frames, dtype=int)
        frames_uint8 = video_reader.get_batch(frame_indices).asnumpy()

    if frames_uint8 is None: raise ValueError("No valid video or image frames to process.")

    visual_feats, audio_len_in_s = _encode_visual_features_safely(frames_uint8, model_dict, fps, batch_size, sync_batch_size, enable_profiling)

    # Use the provided negative prompt, or fall back to a default.
    neg_prompt = negative_prompt if negative_prompt and negative_prompt.strip() else "noisy, harsh"
    
    prompts = [neg_prompt, prompt]
    clap_model = model_dict.clap_model
    try:
        clap_model.to(model_dict.device)
        text_feat_res, _ = encode_text_feat(prompts, model_dict)
    finally:
        clap_model.to("cpu")

    text_feat = text_feat_res[1:]; uncond_text_feat = text_feat_res[:1]
    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = AttributeDict({'text_feat': text_feat, 'uncond_text_feat': uncond_text_feat})
    return visual_feats, text_feats, audio_len_in_s

# In utils.py, add this entire block at the end of the file

from diffusers.utils.torch_utils import randn_tensor
from hunyuanvideo_foley.utils.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

def _retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    """ Helper function forked from the library's pipeline.py """
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _prepare_latents(scheduler, batch_size, num_channels_latents, length, dtype, device):
    """ Helper function forked from the library's pipeline.py """
    shape = (batch_size, num_channels_latents, int(length))
    latents = randn_tensor(shape, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


def denoise_process_safely(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, **kwargs):
    """ Memory-safe fork of denoise_process that ensures all tensors are on the correct device. """
    # The main foley_model's device is managed by accelerate and can be a mix of cpu/cuda
    # We will use the general cuda device for other tensors.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        logger.info("[Denoise] Moving DAC model to device (Foley model is auto-mapped)...")
        model_dict.dac_model = model_dict.dac_model.to(device)
        
        # The main model is bfloat16, this is our target.
        target_dtype = torch.bfloat16

        # Ensure all feature tensors are moved to the primary GPU device
        visual_feats.siglip2_feat = visual_feats.siglip2_feat.to(device)
        visual_feats.syncformer_feat = visual_feats.syncformer_feat.to(device)
        text_feats.text_feat = text_feats.text_feat.to(device)
        text_feats.uncond_text_feat = text_feats.uncond_text_feat.to(device)
        
        guidance_scale = kwargs.get('guidance_scale', 4.5)
        num_inference_steps = kwargs.get('num_inference_steps', 50)
        batch_size = kwargs.get('batch_size', 1)

        scheduler = FlowMatchDiscreteScheduler(
            shift=cfg.diffusion_config.sample_flow_shift,
            reverse=cfg.diffusion_config.flow_reverse,
            solver=cfg.diffusion_config.flow_solver,
            use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
            flux_base_shift=cfg.diffusion_config.flux_base_shift,
            flux_max_shift=cfg.diffusion_config.flux_max_shift,
        )

        timesteps, _ = _retrieve_timesteps(scheduler, num_inference_steps, device)

        latents = _prepare_latents(
            scheduler, batch_size=batch_size,
            num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
            length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
            dtype=target_dtype, device=device,
        )

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising steps"):
            latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_input = scheduler.scale_model_input(latent_input, t)
            t_expand = t.repeat(latent_input.shape[0]).to(device)

            siglip2_feat = visual_feats.siglip2_feat.repeat(batch_size, 1, 1)
            uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat.shape[1])
            siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat], dim=0) if guidance_scale > 1.0 else siglip2_feat

            syncformer_feat = visual_feats.syncformer_feat.repeat(batch_size, 1, 1)
            uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat.shape[1])
            syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat], dim=0) if guidance_scale > 1.0 else syncformer_feat

            text_feat_repeated = text_feats.text_feat.repeat(batch_size, 1, 1)
            uncond_text_feat_repeated = text_feats.uncond_text_feat.repeat(batch_size, 1, 1)
            text_feat_input = torch.cat([uncond_text_feat_repeated, text_feat_repeated], dim=0) if guidance_scale > 1.0 else text_feat_repeated

            # --- FINAL BRUTE-FORCE AUTOCAST FIX ---
            # This forces every operation inside the model call to use bfloat16,
            # which should resolve any lingering dtype mismatches.
            with torch.autocast(device_type=device.type, enabled=True, dtype=target_dtype):
                noise_pred = model_dict.foley_model(
                    x=latent_input.to(device), 
                    t=t_expand, 
                    cond=text_feat_input.to(device),
                    clip_feat=siglip2_feat_input.to(device), 
                    sync_feat=syncformer_feat_input.to(device),
                    return_dict=True,
                )["x"]

            noise_pred = noise_pred.to(dtype=torch.float32)

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents.to(device), return_dict=False)[0].to(latents.device)

        with torch.no_grad():
            audio = model_dict.dac_model.decode(latents.to(model_dict.dac_model.device))
            audio = audio.float().cpu()

        audio = audio[:, :int(audio_len_in_s * model_dict.dac_model.sample_rate)]
        return audio, model_dict.dac_model.sample_rate

    finally:
        logger.info("[Denoise] Offloading DAC model from device...")
        model_dict.dac_model = model_dict.dac_model.to("cpu")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def create_node_exit_values(silent_audio, passthrough_video=None, passthrough_images=None, message="Process skipped or failed."):
    """
    Creates a standardized tuple of return values for exiting a node early.
    Handles passthrough of video or image inputs.
    """
    audio_output = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": 48000} if silent_audio else None
    
    # Prioritize passing through the image tensor if it exists
    if passthrough_images is not None:
        frames_output = passthrough_images
    else:
        frames_output = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        
    # Pass through the video path if it exists
    video_path_output = extract_video_path(passthrough_video) if passthrough_video else ""
    
    return (video_path_output, frames_output, audio_output, message)

def _encode_video_with_siglip2_safely(pixel_values, model_dict):
    """
    A wrapper to handle different versions of the transformers library for SigLIP2.
    """
    if hasattr(model_dict.siglip2_model, 'get_image_features'):
        # Older transformers versions
        return model_dict.siglip2_model.get_image_features(pixel_values=pixel_values)
    else:
        # Newer transformers versions
        return model_dict.siglip2_model(pixel_values=pixel_values).image_embeds

def _batched_frames_to_tensor(pil_list, preprocess_func, batch_size, device):
    """ Preprocesses and moves frames to device in small batches to save VRAM. """
    tensor_chunks = []
    for i in range(0, len(pil_list), batch_size):
        batch_pil = pil_list[i : i + batch_size]
        batch_preprocessed = [preprocess_func(im) for im in batch_pil]
        batch_tensor = torch.stack(batch_preprocessed, dim=0).to(device)
        tensor_chunks.append(batch_tensor)
        # Immediately clear the list to save RAM
        del batch_pil, batch_preprocessed, batch_tensor
    
    # Once all chunks are on the device, concatenate them.
    # This is much more memory-efficient than creating one giant tensor at once.
    full_tensor = torch.cat(tensor_chunks, dim=0).unsqueeze(0)
    del tensor_chunks
    return full_tensor    


@torch.inference_mode()
def encode_video_with_sync(x: torch.Tensor, model_dict, batch_size: int = -1):
    """
    The input video of x is best to be in fps of 24 of greater than 24.
    Input:
        x: tensor in shape of [B, T, C, H, W]
        batch_size: the batch_size for synchformer inference
    """
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size : i * step_size + segment_size])
    x = torch.stack(segments, dim=1).cuda()  # (B, num_segments, segment_size, 3, 224, 224)

    # --- THE DEFINITIVE FIX ---
    # Get a reference to the model and ensure it is on the same device as the data tensor 'x'.
    # This resolves the CPU (weight) vs. GPU (input) conflict.
    sync_model_on_device = model_dict.syncformer_model.to(x.device)
    # --- END FIX ---

    outputs = []
    if batch_size < 0:
        batch_size = b * num_segments
    x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
    for i in range(0, b * num_segments, batch_size):
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.half):
            # Use the model that is now guaranteed to be on the correct device.
            outputs.append(sync_model_on_device(x[i : i + batch_size]))
    x = torch.cat(outputs, dim=0)  # [b * num_segments, 1, 8, 768]
    x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
    return x