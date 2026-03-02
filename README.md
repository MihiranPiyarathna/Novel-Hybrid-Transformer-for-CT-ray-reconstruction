# Novel Hybrid Transformer for CT ray reconstruction
Reconstructing Low Radiation CT Scans using Transformer and UNet Res Based Neural Networks

## Research Problem and Real World impact
```
* 6.7 million CT scans performed in the year ending March 2022. (NHS, England)
* 375 million CT scans, worldwide each year. (UNSCEAR)(Wojcik, 2022)
* ~10% of medical imaging procedures, but 
* ~60% of Human medical radiation exposure (UNSCEAR)(Wojcik, 2022)
* Alternative, Low radiation scans are noisy and often lead to misdiagnosis.
* Our Research Problem is compiling a hybrid NN to reconstruct a clear CT image that
* can be reasonable alternative to a HDCT from Low Radiation DICOM X-ray projections.
```
## 1 Abstract 

Computed tomography from low radiation dose is challenging due to the high noise and artefacts from the projection data. The problem escapes the domain of linear algorithms due to the complexity of noise and the high bar of structural similarity expected in the reconstruction to feasibly make medical diagnosis. Popular NN based attempts include complex and very computationally costly unrolled iterative methods such as Learned Primal Dual and itNet. Our approach is applying a hybrid solution which enhances on the efficiency of the code by implementing a 2 model pipeline, where the first reconstruction, a linear back projection is followed by a learned Neural Network based on the Transformer Architecture. The proposed method surpasses many published models on this problem and surpassed the PSNR threshold of 30 and SSIM of 70%.

![Architecture](https://github.com/user-attachments/assets/a6c461ad-79f9-433c-bdf3-de0030abe2d5)

## 2 Contribution and Thesis Outline

In this paper we propose a novel hybrid Transformer model that achieves benchmark quality performance against established models that are more costly in computation. A shift from the existing approaches of rolled iterations to a two stage approach was implemented due to the nature of the problem, where efficient output is key. 

As regular with the two stage approaches, We run the LD CT sinogram data (X-ray Photon detector measurements) through a classical mathematical reconstruction model based on discretized version of the Inverse Radon transform called Filtered Back Projection – FBP. (Natterer and Wang, 2002). We then performed hyper parameter tuning on this classical model to analyse the best parameters to pass in with our LDCT sinograms.

Then we build upon that impression of the CT scan using Learned methods addressing Gaussian Denoising tasks. We experimented on two leading Gaussian Denoisers, Deep Denoiser Prior by Zhang, K. et al. (Zhang et al., 2022) and Transformer Neural Network first introduced by Zamir, S. et al. (Zamir et al., 2022). 

The strategy is to harness the high level of efficiency and accuracy of either a UNet Residual Network or a Transformer based model to inference a noiseless version as close as to a High radiation Dose CT scan.

Since gaussian denoising relatively very efficient, we could remove the complexity of an unrolled deep learning solution to the research problem as well as get rid of the extensive resources needed to parse a CT sinogram iteratively.
We had to write a helper transition layer to manage the input, output shapes, sizes between the two sub-models, before trying to glue the DRUnet to the FBP implementation.

Then we introduce a second hybrid model with the Restormer Transformer model implemented after the FBP model. We fine tuned 4 instances of this hybrid model using different iterations and different dataset sizes being used to re-train. 

Both the hybrid models achieved benchmark level performance exceeding published works such as ISTA Reconstructor (Liu et al., 2020) and iRadonMAP by He et al. (He, Wang and Ma, 2020). IRadonMAP was particularly interesting due to the fact that it was the highest performing NN in the published model set we used for evaluations. iRadonMAP is a fully learned model based on AUTOMAP architecture (Zhu et al., 2018).

##	3 METHODOLOGY – DESIGN & ANALYSIS


### 3.1	Filtered Back Projection (FBP) - DESIGN
FBP model, (Zhu et al., 2018) Projection of a CT scan is noted as its’ Radon Tranformation. FBP models reconstruct the image from the Sinogram measurements, hence executing an inverse Radon Transformation. We experimented on existing FBP models, did fine tuning on FBP model model parameters based on the Design blueprint noted below.
1.	1D Fourier transform on each ray projection  
<img width="423" height="68" alt="image" src="https://github.com/user-attachments/assets/0f74b365-d67d-4cff-8bfa-53d5f3b7f1db" />

2.	Multiply by a frequency-domain filter - H(ω) (this is the “filter” part). The ideal inverse Radon requires a ramp (|ω|) response — the Ram-Lak / ramp filter — but in practice we usually use a windowed (smoothed/ band-limited) version to control noise and aliasing:  
<img width="417" height="93" alt="image" src="https://github.com/user-attachments/assets/37eb5a39-2934-4874-a8c6-e7464536d285" />

3.	Inverse Fourier transform the filtered projections back to the spatial detector domain and backproject (smear) them across the image for every angle 𝜃 and integrate over angles:  
<img width="305" height="43" alt="image" src="https://github.com/user-attachments/assets/b687e9ce-3dbf-4431-8ff0-2e91a53c88e5" />

4.	Discretely: you perform a 1D FFT on each sinogram row, multiply by the discrete filter response, inverse FFT, and then accumulate (interpolated) values into the image for each projection angle. This restores the high-frequency attenuation introduced by simple backprojection and gives a sharp reconstructed image.
5.	FREQUENCY SCALING parameter exposes a frequency-scaling (or cutoff fraction) parameter d∈ (0,1). It define an effective cutoff frequency 𝜔𝑐=𝑑⋅𝜔𝑁, where 𝜔𝑁 is the Nyquist frequency for the sampled detector (half the sampling rate along 𝑠);  
<img width="316" height="100" alt="image" src="https://github.com/user-attachments/assets/943997e6-f17d-40d6-9e38-3e74bb1d1c42" />

6.	Ran-Lak parameter: maximum resolution (sharp edges) but amplifies noise & aliasing. Use only when SNR is high and sampling is dense.
7.	Hann Parameter (or Hamming / Cosine / Shepp-Logan) = gentler high-frequency roll-off → less noise, fewer ringing artefacts, but slightly lower spatial resolution. Good default for clinical/SNR-limited data. After a grid search and some exploration we decided to use Hann as our default FBP model parameter.
8.	Lower frequency_scaling reduces noise and aliasing further at the cost of resolution. If you see speckly noise / streaks, try lowering the cutoff; if edges are too soft, increase it. Typical values explored in practice: 0.5–1.0 (depends on detector sampling and SNR)

### 3.2	RESTORMER Trasnformer model - DESIGN
We already highlighted the core architecture of the NN in the literature review (Zamir et al., 2022). To expand on other aspects of the model that helped us fine tune and retrain the model, 
1.	Loss function, L₁ loss uses pixelwise L1 between predicted and ground truth images. Concretely the loss used is the standard L₁ (mean / sum over pixels) between output 𝐼^ and ground truth I: 
<img width="191" height="58" alt="image" src="https://github.com/user-attachments/assets/973d92ba-f9b2-41b6-8b6b-1ad80140170d" />

2.	AdamW optimizer (β₁=0.9, β₂=0.999, weight decay 1e-4), initial LR 3e-4 decayed to 1e-6 via cosine annealing; progressive learning on patch sizes (start small + large batch, then increase patch size and reduce batch) is used to help the model learn global image statistics. These choices are relevant because they interact with the pixel loss to produce the reported results.

### 3.3	DRUnet – Unet Res model – DESIGN
We already highlighted the core architecture of the NN in the literature review (Zhang et al., 2022). To expand on the other aspects of the model that helped us fine tune the model,
1.	Training data: large training set assembled from BSD, Waterloo, DIV2K, Flick2K (authors train on a large number of images). Noise is simulated as AWGN with 𝜎 randomly sampled from a training range (so the network learns to denoise at different noise levels). paper reports training with Adam, starting lr 1 x 10**-4 with scheduled halving, batch patches of size 128.

2.	The HQS (Half-Quadratic Splitting) PnP algorithm (DPIR): DPIR uses HQS to split the minimization into alternating data and prior/denoising subproblems. Starting from the MAP energy, HQS introduces an auxiliary variable

## 5	TESTING, VALIDATION AND EVALUATION


Image quality in LDCT is typically measured by PSNR, SSIM, as quantitative metrics, as well as task-specific criteria such as lesion detectability. On these benchmarks, learned methods consistently beat FBP and match or exceed IR. A better Learned NN tend to strike a good balance of PSNR and SSIM. Across studies, a rule of thumb is that a successful DL method raises PSNR by 3–7 dB over FBP/TV, though absolute values depend on the dose level and dataset. 
Importantly, improved metrics correspond to clinically meaningful gains. Authors report that low-dose reconstructions can achieve lesion contrast and visibility on par with standard dose when using modern DL reconstructions. 
Our testing platform was as follows:
	We tested 14 models as below and got leading PSNR values and SSIM values. Testing setup is as follows:
1.	Dataset – Lodopab, Space: ‘Test’
2.	Tested evaluation on 10, 50, 256 and 3,553 test sinogram and their HDCT pairs.
3.	Our hybrid model performance was as below

<img width="776" height="495" alt="image" src="https://github.com/user-attachments/assets/fb06e85f-daae-4c5d-b973-98180c4105b7" />

Figure 6 Hybrid Model Performance
 
 
<img width="1831" height="796" alt="image" src="https://github.com/user-attachments/assets/3a64e909-b348-4f98-8a1e-e9a63c3b8660" />

Figure 7 Hybrid model surpasing generic DRUnet and Iradonmap +
Figure 8 Hybrid Model surpassing learned Iradonmap

In figure 7 pls. note the default pretrained Gaussian Denoiser - DRUnet performing comparatively poorly with our Hybrid model.
Figure 8 denotes the the Hybrid Transformer model exceeding the Inverse Radon Mapping Model (AUTOMAP based) and subsequently achieve >30 PSNR score.
Furthurmore, We compared our method against classical reconstructors below to get more context into the surpassing performance,
1.	CGReconstructor (Conjugate-Gradient Least Squares) (Greif and Wathen, 2019)solving iteratively. CGLS is efficient for well-conditioned problems but can degrade with high noise.

2.	LandweberReconstructor, (Landweber, 1951) a simple gradient-descent iteration. Landweber often acts as a regularization by early stopping.
Each baseline was tuned (e.g. filter type, iteration count) to maximize PSNR/SSIM on validation. This allowed a fair comparison of reconstruction quality.
Evaluation: Reconstructions were quantitatively scored using PSNR and SSIM against the ground truth images, as in the LoDoPaB challenge. We also monitored inference speed and visual artifacts.

### 5.1	Summary of Evaluation
Our hybrid Restormer model achieved substantially higher PSNR and SSIM than classical reconstruction methods and several notable learned methods. For example, the best tuned standalone FBP gave about 24 dB PSNR and 0.58 SSIM on test cases (with Hann filter at 0.8 scaling in our trials). In contrast, the fine-tuned Restormer with annexed FBP with Transformer model typically boosted PSNR by several dB and SSIM closer to 0.70–0.80. These gains match prior reports that DL denoising can significantly improve LDCT metrics. (Selig et al., 2025) found that fine-tuning on LoDoPaB pushed their pipeline to first place in the challenge – achieving the highest SSIM of any method.
 Our results similarly show that the learned Restormer reconstructions have visibly reduced streak artifacts and noise compared to FBP, while preserving anatomical detail (see Figure 6,7,8).

Importantly, the inference is efficient. Our pipeline only requires one FBP (fast analytic step) and one forward pass through the Restormer network. This simplicity contrasts with unrolled iterative networks like ItNet, which loop many times (each time doing a projection and backprojection).
As a result, the total runtime of our method was significantly lower. In the LoDoPaB challenge, it was runtime was highly analysed due to the use case of this research problem.
In our experiments, we observed similar runtimes: for a 512×512 slice, FBP took ~0.5 s on a GPU and the Restormer pass took ~0.3–0.5 s, yielding a total well under 2 s per slice – much faster than typical iterative reconstructions.

### 5.2	Comparison to Iterative Methods

The improved accuracy comes at no major cost to computation. Classical iterative methods like Landweber or CGLS require dozens of iterations to converge, each involving costly forward/backprojections. By contrast, our hybrid network reaches a high-quality result in a single shot. Moreover, the learned network implicitly encodes regularization: it effectively learns complex, non-linear priors from training data. This means it can suppress LDCT noise patterns that simple smoothness priors (used by methods like Landweber) cannot. As theory predicts, Landweber’s simple gradient descent is a crude regularization, whereas our network exploits far richer statistics. Conjugate-gradient (CGLS) solves optimally for noiseless data, but it too struggles with Poisson noise and often needs extra filtering. 

In practice, we found that even a well-tuned CGLS or Landweber reconstructor could not match the PSNR/SSIM of our Restormer-based output.

## 7 REFERENCES

Pls. find a detailed list of references in my paper <here>
Inspired mainly by Restormer: Zamir, S.W. et al. (2022)
