"""Simplical Complex Autoencoder Layer"""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCALayer(torch.nn.Module):
    """Layer of a Simplicial Complex Autoencoder (SCA).

    Implementation of the SCA layer proposed in [HZPMC22]_.

    Notes
    -----
    This is the architecture proposed for complex classification.

    References
    ----------
    .. [HZPMC22] Hajij, Zamzmi, Papamarkou, Maroulas, Cai.
        Simplicial Complex Autoencoder
        https://arxiv.org/pdf/2103.04046.pdf

    Parameters
    ----------
    in_channels_1 : int
        Dimension of first input cell features.
    in_channels_2 : int
        Dimension of second input cell features.
    out_channels : int
        Dimension of output cell features.
    att : bool
        Whether attention is used or not.
    """

    def __init__(
        self,
        in_channels_1,
        in_channels_2,
        out_channels,
        att = False,
    ):
        super().__init__()
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels = out_channels
        self.att = att

        self.conv1 = Conv(
            in_channels=in_channels_1,
            out_channels=out_channels,
            att=att,
        )
        self.conv2 = Conv(
            in_channels=in_channels_2,
            out_channels=out_channels,
            att=att,
        )
        self.aggr1 = Aggregation(
            aggr_func="sum",
            update_func="sigmoid",
        )
        self.aggr2 = Aggregation(
            aggr_func="sum",
            update_func = "sigmoid",
        )
        self.aggr3 = Aggregation(
            aggr_func="mean",
            update_func="sigmoid",
        )

    def reset_parameters(self):
        r"""Reset parameters of each layer"""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def weight_func(self, x):
        r"""Weight function for intra aggregation layer according to [HZPMC22]_."""
        return 1/(1+torch.exp(-x))

    def forward(self, x_1, x_2, neighborhood_1, neighborhood_2):
        r"""Forward pass.

        The forward pass was initially proposed in [HZPMC22]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        Adjacency Message Passing Scheme
        .. math::
            \begin{align*}
                &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}  = M(h_{x}^{t, (r)}, h_{y}^{t, (r)}, att(h_{x}^{t, (r)}, h_{y}^{t, (r)}), x, y, \Theta^t) \qquad \text{where } r'' < r < r'
                &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(r' \rightarrow r)} = M(h_{x}^{t, (r)}, h_{y}^{t, (r')}, att(h_{x}^{t, (r)}, h_{y}^{t, (r')}), x, y, \Theta^t)
                &🟧 \quad m_x^{(r \rightarrow r' \rightarrow r)}  = \text{AGG}\_{y \in \mathcal{L}\_\uparrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
                &🟧 \quad m_x^{(r' \rightarrow r)} = \text{AGG}\_{y \in \mathcal{C}(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(r' \rightarrow r)}
                &🟩 \quad m_x^{(r)}  = \text{AGG}\_{\mathcal{N}\_k \in \mathcal{N}}(m_x^{(k)})
                &🟦 \quad h_{x}^{t+1, (r)} = U(h_x^{t, (r)}, m_{x}^{(r)})
            \end{align*}
        Coadjacency Message Passing Scheme
            \begin{align*}
                &🟥 \quad m_{y \rightarrow x}^{(r \rightarrow r'' \rightarrow r)} = M(h_{x}^{t, (r)}, h_{y}^{t, (r)},att(h_{x}^{t, (r)}, h_{y}^{t, (r)}),x,y,{\Theta^t}) \qquad \text{where } r'' < r < r'
                &🟥 \quad m_{y \rightarrow x}^{(r'' \rightarrow r)} = M(h_{x}^{t, (r)}, h_{y}^{t, (r'')},att(h_{x}^{t, (r)}, h_{y}^{t, (r'')}),x,y,{\Theta^t})
                &🟧 \quad m_x^{(r \rightarrow r)}  = AGG_{y \in \mathcal{L}\_\downarrow(x)} m_{y \rightarrow x}^{(r \rightarrow r)}
                &🟧 \quad m_x^{(r'' \rightarrow r)} = AGG_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(r'' \rightarrow r)}
                &🟩 \quad m_x^{(r)}  = \text{AGG}\_{\mathcal{N}\_k \in \mathcal{N}}(m_x^{(k)})
                &🟦 \quad h_{x}^{t+1, (r)} = U(h_x^{t, (r)}, m_{x}^{(r)})
            \end{align*}
        Homology and Cohomology Message Passing Scheme
                &🟥 \quad m_{y \rightarrow x}^{(r' \rightarrow r)}  = M(h_{x}^{t, (r)}, h_{y}^{t, (r')},att(h_{x}^{t, (r)}, h_{y}^{t, (r')}),x,y,{\Theta^t}) \qquad \text{where } r'' < r < r'
                &🟥 \quad m_{y \rightarrow x}^{(r'' \rightarrow r)} = M(h_{x}^{t, (r)}, h_{y}^{t, (r'')},att(h_{x}^{t, (r)}, h_{y}^{t, (r'')}),x,y,{\Theta^t})
                &🟧 \quad m_x^{(r' \rightarrow r)} = AGG_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(r' \rightarrow r)}
                &🟧 \quad m_x^{(r'' \rightarrow r)}  = AGG_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(r'' \rightarrow r)}
                &🟩 \quad m_x^{(r)}  = AGG_{\mathcal{N}\_k \in \mathcal{N}}(m_x^{(k)})
                &🟦 \quad h_{x}^{t+1, (r)} = U(h_x^{t, (r)}, m_{x}^{(r)})

        References
        ----------
        .. [HZPMC22] Hajij, Zamzmi, Papamarkou, Maroulas, Cai.
            Simplicial Complex Autoencoder
            https://arxiv.org/pdf/2103.04046.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_1: torch.Tensor, shape = (n_kchains, in_channels_1)
            Input features of each kchain on the simplicial complex.
        x_2: torch.Tensor, shape = (n_lchains, in_channels_2)
            Input feature of each lchain on the simplicial complex.
        neighborhood_1: torch.sparse, shape = [kchains, mchains]
            Neighborhood matrix mapping input 1 to output.
        neighborhood_2: torch.sparse, shape = [lchains, mchains]
            Neighborhood matrix mapping input 2 to output.

        Returns
        ------- 
        _: torch.Tensor, shape=[n_mchains, channels]
            Output features on output chains. (nodes: 0-chain, edges: 1-chain,...).
        """

        x_1 = self.conv1(x_1, neighborhood_1)
        x_1_list = list(torch.split(x_1, 1, dim=0))
        x_1_weight = self.aggr1(x_1_list)
        x_1_weight = torch.matmul(torch.relu(x_1_weight), x_1.transpose(1, 0))
        x_1_weight = self.weight_func(x_1_weight)
        x_1 = x_1_weight.transpose(1, 0)*x_1

        x_2 = self.conv2(x_2, neighborhood_2, x_1)
        x_2_list = list(torch.split(x_2, 1, dim=0))
        x_2_weight = self.aggr2(x_2_list)
        x_2_weight = torch.matmul(torch.relu(x_2_weight), x_2.transpose(1, 0))
        x_2_weight = self.weight_func(x_2_weight)
        x_2 = x_2_weight.transpose(1, 0)*x_2

        xf = self.aggr3([x_1, x_2])

        return xf
