<div align="center">
<h1> Deep learning-assisted end-to-end metalens imaging </h1>
<h3> DRMI : DNN-based image Reconstruction framework customized for Metalens Imaging system </h3>

Joonhyuk Seo<sup>1,✢</sup>,
Jaegang Jo<sup>2,✢</sup>,
[Joohoon Kim](https://scholar.google.com/citations?hl=en&user=tRNVtewAAAAJ)<sup>3,✢</sup>,
Joonho Kang<sup>2,</sup>, 
[Haejun Chung](https://scholar.google.com/citations?user=O-oZnIwAAAAJ)<sup>2,📧</sup>,
[Junsuk Rho](https://scholar.google.com/citations?user=jdNQRH8AAAAJ)<sup>3,📧</sup>,
[Jehyung Hong](https://scholar.google.com/citations?user=7axCcBkAAAAJ)<sup>2,📧</sup>,

<sup>1</sup> Department of Artificial Intelligence, Hanyang University\
<sup>2</sup> Department of Electronic Engineering, Hanyang University\
<sup>3</sup> Department of Mechanical Engineering, Pohang University of Science and Technology (POSTECH)

(✢) Equal contribution.
(📧) corresponding author.

<hr />

> **Abstract:** *Recent advances in metasurface lenses (metalenses) show great potential for opening a new era of compact imaging, photography, LiDAR, and VR/AR applications. However, the reported performances of manufactured broadband metalenses are still limited due to a fundamental trade-off between broadband focusing efficiency and operating bandwidth, resulting in chromatic aberrations, angular aberrations, and relatively low efficiency. Here, we demonstrate a deep learning-based image restoration framework to overcome these limitations and to realize end-to-end metalens imaging. The proposed image restoration framework achieves aberration-free full-color imaging for one of the largest mass-produced metalens (10-mm-diameter). The metalens imaging assisted by the neural network provides competitive image qualities compared to the ground truth.* 
<hr />
</div>
## Introduction

![](figures/Fig1.png)

Metalenses, ultra-thin film lenses composed of subwavelength structures, have been spotlighted as a technology to overcome the limitations of conventional lenses. However, recent studies suggest that large-area broadband metalenses may suffer from a fundamental trade-off between broadband focusing efficiency and their diameter. Consequently, at present, reported broadband metalenses show chromatic aberration or low focusing efficiency over the large bandwidth, which hinders a commercialization of metalens-based compact imaging. In addition, meta-atom-based metalenses show a narrow Field of View (FoV) due to the angular dispersion of the meta-atoms where meta-atoms are periodically aligned basis of the metasurfaces for simplifying a design process of metasurfaces while maintaining the predicted efficiency.

In this study, we propose the DNN-based image Reconstruction framework customized for Metalens Imaging system (DRMI) to overcome all these physical constraints by learning defects of the largest mass-produced metalenses (a 10-mm diameter). Unlike conventional image restoration tasks, our mass-produced metalens simultaneously undergoes significant chromatic and angular aberrations. Therefore, the level of the metalens image restoration becomes a challenging task. DRMI captures and restores the aberration problems of the metalenses by a data-driven learning-based approach. Specifically, we collect a few hundred blur images taken from the metalens. Then, DRMI is trained using the images containing the physical defects of the metalens, which dramatically restores the quality of the image created by the compact metalens imaging system. DRMI consists of a Nonlinear Activation Free Network (NAFNet) as the baseline and an adversarial learning scheme in the frequency domain. By applying DRMI to the mass-produced metalens, we construct a hybrid imaging system that achieves high-quality compact imaging. This system is scalable even to a larger aperture and different wavelengths, thereby providing an ultimate solution for a novel miniaturized imaging scheme.
