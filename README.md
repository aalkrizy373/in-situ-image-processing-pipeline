# Computational omics pipeline for analyzing Image-based transcriptomics

# Intro: 
High resolution molecular imaging with high accuracy is a growing challenge as it can be hindered by sizes of large datasets, limits in optical resolution and density of transcripts in single cells. These python programs are designed to help in the method for optimizing the image processing for such in-situ genomic data. 

# Multi-Channel-Deconvolution: 
This program is responsible for deconvoluting very large images that don't fit GPU ram requirements. It opens images, breaks the images into manageable chunks to fit into GPU ram requirements, performs deconvolution and stitches the images back together.

# Alignment-Multi-Rounds: 
This program is responsible for automating the alignment process of large microscopy images that were being aligned manually. The script aligns to construct a comprehensive three dimensional map, while simultaneously performing standardized data transformations, visualization and regional annotation. It aligns multiple images of the same spatial location across multiple rounds of images. 

# Multi-Round-Pixel-Decoder: 
This algorithm will be used for identifying transcripts that display spatial expression patterns. It uses the starfish package. The program requires customizing accurate identification of transcript boundaries. It localizes and decodes transcript molecules, finding spots by fitting the local intensity maxima. It then decodes every pixel and combines these pixel values to map spots/gene targets with an expected intensity using the skimage.measure.label() and skimage.measure.regionprops(), which also gives back spot locations and attributes.

While this pixel-appraoch method is meant to accurately detect spots which can be advantageous when working with dense data. However in our data, it also ended up decoding noise in images, which was addressed by incorporating a size threshold for the spots/gene targets in the images, labeled in the program as the "min_area" and "max_area". 

The decoded data is saved in a decoded-intensity-table, which showcases the outputs of the ImageStack in a table with spot locations, intensities, and sizes. The program also uses a codebook which can be customized based on experiment. It contains maps of expected intensities across multiple image rounds to the spot/gene targets that are encoded and mapping of channels to the integer indices that are used to represent them.
