"""Shape and causality checks for the Stage 1 dense continuous autoencoder.

These do not test speech quality (that requires real data and training) —
only that the architecture is wired correctly and that no module leaks
information from future samples into the current output.
"""

import torch

from aevum.models.autoencoder import DenseContinuousAutoencoder
from aevum.models.decoder import ContinuousDecoder
from aevum.models.encoder import ContinuousEncoder
from aevum.models.frontend import CausalAcousticFrontend
from aevum.models.generator import CausalWaveformGenerator

SAMPLE_RATE = 24_000


def test_frontend_total_stride_is_240() -> None:
    frontend = CausalAcousticFrontend()
    assert frontend.total_stride == 240


def test_generator_total_stride_is_240() -> None:
    generator = CausalWaveformGenerator()
    assert generator.total_stride == 240


def test_encoder_output_shape() -> None:
    encoder = ContinuousEncoder(latent_dim=512)
    waveform = torch.randn(2, 1, 240 * 50)  # 500 ms
    z, _ = encoder(waveform)
    assert z.shape == (2, 50, 512)


def test_decoder_output_shape() -> None:
    decoder = ContinuousDecoder(event_dim=512, output_dim=384)
    u = torch.randn(2, 50, 512)
    y, _ = decoder(u)
    assert y.shape == (2, 50, 384)


def test_autoencoder_end_to_end_shape() -> None:
    model = DenseContinuousAutoencoder()
    num_frames = 25  # 250 ms
    waveform = torch.randn(2, 1, model.total_stride * num_frames)
    reconstructed = model(waveform)
    assert reconstructed.shape[0] == 2
    assert reconstructed.shape[1] == 1
    assert reconstructed.shape[2] == model.total_stride * num_frames


def test_frontend_is_causal() -> None:
    """Changing a sample must not affect frontend output at or before that sample's frame."""
    frontend = CausalAcousticFrontend()
    frontend.eval()

    torch.manual_seed(0)
    waveform = torch.randn(1, 1, 240 * 10)
    perturbed = waveform.clone()
    perturbed[:, :, 240 * 5 :] += 10.0  # perturb everything from frame 5 onward

    with torch.no_grad():
        out_a = frontend(waveform)
        out_b = frontend(perturbed)

    # Frames strictly before the perturbation must be identical.
    assert torch.allclose(out_a[:, :, :5], out_b[:, :, :5], atol=1e-5)
    # Something after the perturbation must have changed, otherwise this test is vacuous.
    assert not torch.allclose(out_a[:, :, 5:], out_b[:, :, 5:], atol=1e-5)


def test_generator_is_causal() -> None:
    generator = CausalWaveformGenerator(input_dim=384)
    generator.eval()

    torch.manual_seed(0)
    y = torch.randn(1, 384, 10)
    perturbed = y.clone()
    perturbed[:, :, 5:] += 10.0

    with torch.no_grad():
        out_a = generator(y)
        out_b = generator(perturbed)

    boundary = 5 * generator.total_stride
    assert torch.allclose(out_a[:, :, :boundary], out_b[:, :, :boundary], atol=1e-5)
    assert not torch.allclose(out_a[:, :, boundary:], out_b[:, :, boundary:], atol=1e-5)
