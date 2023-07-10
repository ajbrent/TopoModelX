"""Test the HSN layer."""

import torch

from topomodelx.nn.simplicial.sca_layer import SCALayer


class TestSCALayer:
    """Test the HSN layer."""

    def test_amps_forward(self):
        """Test the forward pass of the SCA layer using AMPS."""
        up_channels = 7
        out_channels = 5
        n_up_chains = 20
        n_out_chains = 15
        lap_up = torch.randint(0, 2, (n_out_chains, n_out_chains)).float()
        incidence_matrix_up = torch.randint(0, 2, (n_out_chains, n_up_chains)).float()

        x_1 = torch.randn(n_out_chains, out_channels)
        x_2 = torch.randn(n_up_chains, up_channels)
        sca = SCALayer(out_channels, up_channels, out_channels, att=False)
        output = sca.forward(x_1, x_2, lap_up, incidence_matrix_up)

        assert output.shape == (n_out_chains, out_channels)

    def test_cmps_forward(self):
        """Test the forward pass of the SCA layer using CMPS."""
        down_channels = 4
        out_channels = 5
        n_down_chains = 10
        n_out_chains = 15
        lap_down = torch.randint(0, 2, (n_out_chains, n_out_chains)).float()
        incidence_matrix = torch.randint(0, 2, (n_down_chains, n_out_chains)).float()
        incidence_transpose = incidence_matrix.transpose(1, 0)

        x_1 = torch.randn(n_out_chains, out_channels)
        x_2 = torch.randn(n_down_chains, down_channels)
        sca = SCALayer(out_channels, down_channels, out_channels, att=False)
        output = sca.forward(x_1, x_2, lap_down, incidence_transpose)

        assert output.shape == (n_out_chains, out_channels)

    def test_hcmps_forward(self):
        """Test the forward pass of the SCA layer using HCMPS."""
        up_channels = 7
        down_channels = 4
        out_channels = 5
        n_down_chains = 10
        n_up_chains = 20
        n_out_chains = 15
        incidence_matrix = torch.randint(0, 2, (n_down_chains, n_out_chains)).float()
        incidence_matrix_up = torch.randint(0, 2, (n_out_chains, n_up_chains)).float()

        x_1 = torch.randn(n_down_chains, down_channels)
        x_2 = torch.randn(n_up_chains, up_channels)
        sca = SCALayer(down_channels, up_channels, out_channels, att=False)
        output = sca.forward(x_1, x_2, incidence_matrix.transpose(1, 0), incidence_matrix_up)

        assert output.shape == (n_out_chains, out_channels)


    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        sca = SCALayer(channels, channels, channels)
        sca.reset_parameters()

        for module in sca.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
