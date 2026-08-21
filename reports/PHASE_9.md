# PHASE 9 — Vision, end to end

Status: **images work through the OpenAI API**, verified by the model answering
correctly about content and spatial position, with prefix caching and
speculation still live.

```
./build/cuda_server --model $MODEL --vision --max-context 8192
```

A 448x448 PNG with a red circle at upper-left and a blue square at lower-right,
sent as a `data:` URL:

> The image contains two simple geometric shapes on a white background:
> 1. **A red circle** located in the upper-left portion of the image.
> 2. **A blue square** located in the lower-right portion of the image.

Both the colours and the positions are right, which is the check that matters:
getting the patch order, the position-embedding resample, the mrope box or the
splice wrong all produce an answer that is confidently about the wrong picture.

| | measured |
|---|---|
| 448x448 image | 784 patches -> **196 image tokens** |
| prompt with image | 248 tokens, prefill 512 tok/s |
| decode | 45.4 tok/s |
| second turn, same image | **304 of 320 tokens cached**, prefill 5650 tok/s |
| tower resident | 0.858 GiB = **28,114 tokens of FP8 KV** |

---

## What the tower needed

Verified against the official `Qwen3_5VisionModel` (`tools/dump_vision_ref.py`):

| stage | rel error | |
|---|---|---|
| patch_embed + resampled pos_embed | 1.96e-03 | OK |
| image tokens (merger output) | 2.70e-02 | OK |
| mrope positions | 0 / 228 entries differ | OK |

Four things are not guessable from the shapes, and each produces a plausible
wrong answer rather than an obvious failure:

1. **`deepstack_visual_indexes` is empty.** Qwen-VL normally injects visual
   features at several language-model layers. Here it does not, so the tower's
   output is spliced in at the embedding layer and nowhere else. That removes
   the hardest part of the architecture.
2. **Patches are in spatial-merge-block order**, not raster:
   `permute(0, 2, 5, 3, 6, 1, 4, 7)` in the reference processor. That is what
   makes the 2x2 merge a pure reshape at the end.
3. **`interpolation_align_corners` is TRUE.** The learned 48x48 pos_embed is
   bilinearly resampled with the endpoint-matching form, not the half-pixel
   form. I wrote it the other way first; the tight `patch+pos_embed` tap caught
   it in one run instead of leaving a half-cell offset smeared through 27 blocks.
4. **Two different GELUs in one tower.** The block MLP uses `gelu_pytorch_tanh`;
   the merger uses `nn.GELU()`, the exact erf one.

## mrope: where the positions stop being sequential

A text token advances all three rope axes by one. An image does not:

```
image token j :  t = cur + j/(H*W),  h = cur + (j/W)%H,  w = cur + j%W
afterwards    :  cur += max(H, W)          <-- NOT the token count
```

So an image of 196 tokens advances the position by 14, and every token after it
sits 182 positions earlier than its sequence index. transformers calls the gap
`mrope_position_deltas`; decode then runs at `kv_slot + delta`.

That forced a small change in the decode path: **the KV slot and the rope
position are no longer the same number.** `d_step[1]` is the slot, `d_step[3]` is
the rope position, and they are equal only while no image is in the context. The
CUDA graph reads both from device memory, so this costs nothing at replay time.

`gate_vision` checks the whole thing against the reference's own
`get_rope_index`, driven through a stub `self` so no 27B model has to be built.

## The prefix cache trap

`<|image_pad|>` carries no image content. Two different pictures with the same
grid tokenise to **exactly the same ids**, so a prefix cache keyed on tokens
would happily serve one image's KV for another — silently, and with a confident
wrong answer.

The cache key is therefore not the token sequence: each image span is replaced
by an FNV hash of the image bytes, mixed per position. Text tokens are unchanged,
so ordinary prefix reuse is unaffected, and two different images can never share
a prefix. Measured on a second turn with the same image: 304 of 320 tokens
reused, and the model answered about the square from KV it never recomputed.

## Preprocessing

`src/vision/image.cpp`, using stb_image for decode:

* `smart_resize` to a multiple of `patch_size * spatial_merge` within
  `[min_pixels, max_pixels]` from `preprocessor_config.json`,
* separable bicubic resize with the Keys kernel (a = -0.75) and **antialiasing**
  — when downscaling the kernel is stretched by 1/scale so each output pixel
  averages every input pixel it covers. Without that, downscaling aliases and
  the tower sees a different image from the one the reference saw,
* normalise with mean 0.5 / std 0.5, i.e. to [-1, 1],
* patchify into merge-block order with each patch laid out
  `[channel][temporal][py][px]`, the still image repeated across the temporal
  axis.

**Not gated against PIL.** The ViT is checked against the reference on synthetic
`pixel_values`, which deliberately separates "does the tower compute the right
thing" from "does our resize match PIL bit for bit". The second is not verified,
and small resize differences are the most likely source of any disagreement with
transformers on a real photograph.

## Limits

* Images arrive as `data:` URLs or local paths. Remote URLs are **not** fetched:
  that would make the server issue outbound requests, which it does not do.
* Video is rejected. The tower has the temporal axis and the code paths are
  shaped for it, but nothing exercises it.
* `--vision-max-patches` defaults to 4096, i.e. 1024 image tokens.
* The vision tower is bf16 and unquantised, as it ships. INT8 would halve the
  0.858 GiB, which is the obvious move if vision and a 262K context ever have to
  coexist -- at 262144 there is currently no room for both.
