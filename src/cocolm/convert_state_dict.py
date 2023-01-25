# Copyright (c) Microsoft Corporation.
# The script is largely adapted from the huggingface transformers library
# and https://github.com/microsoft/COCO-LM library

import torch
import logging

from transformers.modeling_utils import cached_file, WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME

logger = logging.getLogger(__name__)


def get_checkpoint_from_transformer_cache(
        archive_file, pretrained_model_name_or_path, pretrained_model_archive_map,
        cache_dir, force_download, proxies, resume_download,
):
    import pdb; pdb.set_trace()
    try:
        resolved_archive_file = cached_file(pretrained_model_name_or_path, filename=archive_file, 
                                            cache_dir=cache_dir, force_download=force_download, 
                                            proxies=proxies, resume_download=resume_download)
    except EnvironmentError:
        if pretrained_model_name_or_path in pretrained_model_archive_map:
            msg = "Couldn't reach server at '{}' to download pretrained weights.".format(
                archive_file)
        else:
            msg = "Model name '{}' was not found in model name list ({}). " \
                  "We assumed '{}' was a path or url to model weight files named one of {} but " \
                  "couldn't find any such file at this path or url.".format(
                pretrained_model_name_or_path,
                ', '.join(pretrained_model_archive_map.keys()),
                archive_file,
                [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME])
        raise EnvironmentError(msg)

    if resolved_archive_file == archive_file:
        logger.info("loading weights file {}".format(archive_file))
    else:
        logger.info("loading weights file {} from cache at {}".format(
            archive_file, resolved_archive_file))

    return torch.load(resolved_archive_file, map_location='cpu')