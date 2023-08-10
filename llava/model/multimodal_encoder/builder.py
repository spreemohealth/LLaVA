from .clip_encoder import CLIPVisionTower, BiomedClip

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("microsoft/BiomedCLIP"):
        return BiomedClip('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    

    raise ValueError(f'Unknown vision tower: {vision_tower}')
