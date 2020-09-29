# Detection of Malicious Spatial Domain Steganography over Noisy Channels Using Convolutional Neural Networks

### Abstract :
Image steganography can have legitimate uses, for example, augmenting an image with a watermark for copyright reasons, but can also be utilized for malicious purposes. We investigate the detection of malicious steganography using neural networkbased classification when images are transmitted through a noisy channel. Noise makes detection harder because the classifier must not only detect perturbations in the image but also decide whether they are due to the malicious steganographic modifications or due to natural noise. Our results show that reliable detection is possible even for state-of-the-art steganographic algorithms that insert stego bits not affecting an image’s visual quality. The detection accuracy is high (above 85%) if the payload, or the amount of the steganographic content in an image, exceeds a certain threshold. At the same time, noise critically affects the steganographic information being transmitted, both through desynchronization (destruction of information which bits of the image contain steganographic information) and by flipping these bits themselves. This will force the adversary to use a redundant encoding with a substantial number of error-correction bits for reliable transmission, making detection feasible even for small payloads. ([Link to paper](https://www.ingentaconnect.com/contentone/ist/ei/pre-prints/content-ei2020-mwsf-076?crawler=true&mimetype=application/pdf))

### Block Diagram :
![block_diagram](https://user-images.githubusercontent.com/61373911/89793382-196c7580-db26-11ea-94be-875a5763bf0c.PNG)

### Citation :
@article{prasad2020detection,
  title={Detection of malicious spatial-domain steganography over noisy channels using convolutional neural networks},
  author={Prasad, Swaroop Shankar and Hadar, Ofer and Polian, Ilia},
  journal={Electronic Imaging},
  volume={2020},
  number={4},
  pages={76--1},
  year={2020},
  publisher={Society for Imaging Science and Technology}
}
