# Atomically-accurate-de-novo-design-of-single-domain-antibodies
## Essay Summery

RFdiffusion uses the AlphaFold214/RF2 frame representation of protein backbones comprising the Cɑ coordinate and N-Cɑ-C rigid orientation for each residue. During training, a noising schedule is used that, over a set number of “timesteps” (T), corrupts the protein frames to distributions indistinguishable from random distributions (Cɑ coordinates are corrupted with 3D Gaussian noise, and residue orientations with Brownian motion on SO3). During training, a PDB structure and a random timestep (t) are sampled, and t noising steps are applied to the structure. RFdiffusion predicts the de-noised (pX0) structure at each timestep, and a mean squared error (m.s.e.) loss is minimized between the true structure (X0) and the prediction. At inference time, translations are sampled from the 3D Gaussian and uniform rotational distributions (XT) and RFdiffusion iteratively de-noises these frames to generate a new protein structure.

To explore the design of antibodies, we fine-tuned RFdiffusion predominantly on antibody complex structures (Fig. 1; Methods). At each step of training, an antibody complex structure is sampled, along with a random timestep (t), and this number of noise steps are added to corrupt the antibody structure (but not the target structure). To permit specification of the framework structure and sequence at inference time, the framework sequence and structure are provided to RFdiffusion during training (Fig. 1B). Because it is desirable for the rigid body position (dock) between antibody and target to be designed by RFdiffusion along with the CDR loop conformations, the framework structure is provided in a global-frame-invariant manner during training (Fig. 1C). We utilize the “template track” of RF/RFdiffusion to provide the framework structure as a 2D matrix of pairwise distances and dihedral angles between each pair of residues (a representation from which 3D structures can be accurately recapitulated)15, (Extended Data Fig. 1A). The framework and target templates specify the internal structure of each protein chain, but not their relative positions in 3D space (in this work we keep the sequence and structure of the framework region fixed, and focus on the design solely of the CDRs and the overall rigid body placement of the antibody against the target). In vanilla RFdiffusion, de novo binders can be targeted to specific epitopes at inference time through training with an additional one-hot encoded “hotspot” feature, which provides some fraction of the residues the designed binder should interact with. For antibody design, where we seek CDR-loop-mediated interactions, we adapt this feature to specify residues on the target protein with which CDR loops interact (Fig. 1D).

(RFdiffusion 使用蛋白质骨架的 AlphaFold214/RF2 框架表示，包括每个残基的 Cɑ 坐标和 N-Cɑ-C 刚性方向。 在训练期间，使用噪声计划，在一定数量的“时间步长”(T) 上，将蛋白质框架破坏为与随机分布无法区分的分布（Cɑ 坐标被 3D 高斯噪声破坏，残基方向被 SO3 上的布朗运动破坏） ）。 在训练期间，对 PDB 结构和随机时间步 (t) 进行采样，并将 t 个噪声步骤应用于该结构。 RFdiffusion 在每个时间步预测去噪 (pX0) 结构，并最小化真实结构 (X0) 和预测之间的均方误差 (m.s.e.) 损失。 在推理时，从 3D 高斯分布和均匀旋转分布 (XT) 中采样翻译，RFdiffusion 迭代地对这些帧进行去噪以生成新的蛋白质结构。




为了探索抗体的设计，我们主要在抗体复合物结构上微调射频扩散（图 1；方法）。 在训练的每个步骤中，都会对抗体复合物结构以及随机时间步 (t) 进行采样，并添加此数量的噪声步以破坏抗体结构（但不会破坏目标结构）。 为了允许在推理时指定框架结构和序列，在训练期间向 RFdiffusion 提供框架序列和结构（图 1B）。 由于需要通过 RFdiffusion 以及 CDR 环构象来设计抗体和靶标之间的刚体位置（对接），因此在训练期间以全局框架不变的方式提供框架结构（图 1C）。 我们利用 RF/RFdiffusion 的“模板轨迹”来提供框架结构，作为每对残基之间的成对距离和二面角的 2D 矩阵（可以准确概括 3D 结构的表示形式）15，（扩展数据图 1）。 1A）。 框架和目标模板指定了每个蛋白质链的内部结构，但不指定它们在 3D 空间中的相对位置（在这项工作中，我们保持框架区的序列和结构固定，并仅关注 CDR 和整体的设计） 针对目标的抗体的刚体放置）。 在普通 RFdiffusion 中，通过使用额外的单热点编码“热点”特征进行训练，可以在推理时将 de novo 结合物靶向特定表位，该特征提供了设计的结合物应与之相互作用的残基的一部分。 对于抗体设计，我们寻求 CDR 环介导的相互作用，我们调整此特征来指定与 CDR 环相互作用的目标蛋白上的残基.)

### **Fine-tuning RoseTTAFold2 for antibody design validation**

Design pipelines typically produce a wide range of solutions to any given design challenge, and hence readily computable metrics for selecting which designs to experimentally characterize play an important role. An effective way to filter designed proteins and interfaces is based on the similarity of the design model structure to the AlphaFold2 predicted structure for the designed sequence (this is often referred to as "self-consistency"), which has been shown to correlate well with experimental success[**16**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#ref-16),[**17**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#ref-17). In the case of antibodies, however, AlphaFold2 fails to routinely predict antibody-antigen structures accurately[**18**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#ref-18), preventing its use as a filter in an antibody design pipeline.

We sought to build an improved filter by fine-tuning the RoseTTAFold2 structure prediction network on antibody structures. To make the problem more tractable, we provide information during training about the structure of the target and the location of the target epitope to which the antibody binds; the fine-tuned RF2 must still correctly model the CDRs and find the correct orientation of the antibody against the targeted region. With this training regimen, RF2 is able to robustly distinguish true antibody-antigen pairs from decoy pairs and often accurately predicts antibody-antigen complex structures. Accuracy is higher when the bound (holo) conformation of the target structure is provided ([**Extended Data Fig. 2**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#F5)); this is available when evaluating design models, but not available in the general antibody-antigen structure prediction case. At monomer prediction, the fine-tuned RF2 outperforms the previously published IgFold network (which can only model antibody monomer structures)[**19**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#ref-19), especially at CDR H3 structure prediction ([**Extended Data Fig. 3**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#F6)).

When this fine-tuned RF2 network is used to re-predict the structure of RFdiffusion-designed VHHs, a significant fraction are confidently predicted to bind in an almost identical manner to the designed structure ([**Extended Data Fig. 4A**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#F7)). Further, in silico cross-reactivity analyses demonstrate that RFdiffusion-designed VHHs are rarely predicted to bind to unrelated proteins ([**Extended Data Fig. 4B**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#F7)). VHHs that are confidently predicted to bind their designed target are predicted to form high quality interfaces, as measured by Rosetta ddG ([**Extended Data Fig. 4C**](https://www.biorxiv.org/content/10.1101/2024.03.14.585103v1.full#F7)). The fact that many of the designed sequences generated by our RFdiffusion antibody design pipeline are predicted by RF2 to adopt the designed structures and binding modes suggested that RF2 filtering might enrich for experimentally successful binders.

（微调 RoseTTAFold2 以进行抗体设计验证

设计流程通常会针对任何给定的设计挑战产生各种解决方案，因此用于选择要进行实验表征的设计的易于计算的指标发挥着重要作用。 过滤设计蛋白质和界面的有效方法是基于设计模型结构与设计序列的 AlphaFold2 预测结构的相似性（这通常称为“自一致性”），这已被证明与 实验成功16,17。 然而，就抗体而言，AlphaFold2 通常无法准确预测抗体-抗原结构18，从而妨碍其在抗体设计流程中用作过滤器。

我们试图通过对抗体结构上的 RoseTTAFold2 结构预测网络进行微调来构建改进的过滤器。 为了使问题更容易处理，我们在训练期间提供有关目标结构和抗体结合的目标表位位置的信息； 微调后的 RF2 仍必须正确建模 CDR，并找到针对目标区域的抗体的正确方向。 通过这种训练方案，RF2 能够稳健地区分真正的抗体-抗原对和诱饵对，并且通常能够准确预测抗体-抗原复合物结构。 当提供目标结构的结合（全息）构象时，准确性更高（扩展数据图 2）； 这在评估设计模型时可用，但在一般抗体-抗原结构预测情况下不可用。 在单体预测方面，经过微调的 RF2 优于之前发布的 IgFold 网络（只能对抗体单体结构进行建模）19，特别是在 CDR H3 结构预测方面（扩展数据图 3）。

当这个微调的 RF2 网络用于重新预测 RFdiffusion 设计的 VHH 的结构时，可以自信地预测很大一部分会以几乎相同的方式与设计的结构结合（扩展数据图 4A）。 此外，计算机交叉反应性分析表明，很少预测 RFdiffusion 设计的 VHH 会与不相关的蛋白质结合（扩展数据图 4B）。 根据 Rosetta ddG 的测量（扩展数据图 4C），预计 VHH 能够与其设计的靶标结合，从而形成高质量的界面。 事实上，我们的 RFdiffusion 抗体设计流程生成的许多设计序列均由 RF2 预测采用设计的结构和结合模式，这表明 RF2 过滤可能会丰富实验上成功的结合物。）

RFdiffusion经过基因组打乱增加噪声训练后，根据自定义需求预测得到的蛋白质结构需要经过RoseTTAFold2进行验证。


## Coding

Links: https://github.com/RosettaCommons/RFdiffusion

To get started using RFdiffusion, clone the repo:

git clone https://github.com/RosettaCommons/RFdiffusion.git

then need to download the model weights into the RFDiffusion directory.

cd RFdiffusion
mkdir models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

Optional:
wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt

# original structure prediction weights
wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt

Alternatively, I recommend using the link to manually download the weights file.

Next:
cd RFdiffusion

conda env create -f env/SE3nv.yml

conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # change into the root directory of the repository
pip install -e . # install the rfdiffusion module from the root of the repository
