# AiPainter

## A COMPLETE AI PORTRAIT PAINTING SYSTEM
MSc AI Project by Jon Thum.

	•	Style paintings generated by 5 styleGAN2 models trained on Wikiart database.
	•	Style transferred to any uploaded portrait photograph using Style Transfer.
	•	Resultant paintings are created entirely by AI.
	•	Interactive UI for adjusting parameters intuitively.
	•	Filtering of 5000 different style paintings according to image characteristics for quick selection.
	•	Style switching for style paintings - see the same content painted in a different style. 
	•	Abstraction control for final portrait paintings.
	•	Automatic mask creation to preserve facial features in final portrait painting, aiding recognisability.
	•	Also works for non-portrait photographs.
	•	High resolution output of up to 4K (4096x4096 pixels), large enough to print and frame.


## INSTRUCTIONS 
	
	•	Code runs on Colab.
	•	Chrome is recommended browser. Some issues have been noted with Colab on Safari. 
	•	Go to https://colab.research.google.com/ and choose ‘File’ / ‘Open notebook’.
	•	Choose ‘GitHub’ and enter URL  https://github.com/jonthum or search ‘jonthum’.
	•	Choose repository ‘AiPainter’ and open file ‘AiPainter.ipynb’.

## USER GUIDE 

### USING FORMS ON COLAB

	•	Forms allow the UI to be displayed with the code hidden underneath.
	•	To switch from UI to code or vice versa, double click in the space next to the cell title (e.g. AI PAINTER).
	•	Alternatively from the menu click ‘Edit’ /  ‘Show/hide code’.

### USING UI WIDGETS

	•	Drag sliders to change values or click on slider button and use keyboard left and right arrows to step through.
	•	You can enter numbers into the value boxes, but must press return on the keyboard to update them.

### AI PAINTER UI

	•	All sliders and buttons are interactive. Wait a second or two for image to refresh.
	•	Select MODE to ‘ID’ for stepping through all paintings with ID# and changing style for same content (style-switching).
	•	Select MODE to ‘FILTER’ for filtering images within filter ranges. (Note: style-switching does not work in this mode).
	•	In filter mode, step though filtered images with FILTER# (Note: button is blue when active, white when inactive).
	•	Choose ‘1K’, ‘2K’, ‘3K’ or ‘4K’ resolution to save chosen painting to ‘images/results’ directory.

### STYLE TRANSFER UI

	•	Load content image from dropdown menu. Wait a second or two for content image to refresh.
	•	Sliders are not interactive. Adjust sliders then click PROCESS.
	•	Mask window will show automatic mask detection when processing. Select features required for masking from list.
	•	Choose ‘1K’, ‘2K’, ‘3K’ or ‘4K’ resolution to save chosen painting to ‘images/results’ directory.


## UI PARAMETERS

### AI PAINTER 

	•	Experimental: determines how experimental the painting is. A value of 0.7 represents standard conformity to dataset.
	•	Randomise: rearranges content in image (from Random dropdown select ON then click Randomise button).

### STYLE TRANSFER 

	•	Abstraction: determines amount of abstraction in the painting. Values near 0 are more photo-real, values near 1 more abstract.
	•	Tex Scale: determines the scale of style texture applied to the content image. Higher values will produce more texture detail.
	•	Tex Strength: determines the strength of the style texture applied to the content image. Higher values will produce more texture.
	•	Preservation: determines the amount of preservation applied to the mask. Higher values will retain more of the content features.
	•	Original Col: allows mixing back to the original colour of the content image, keeping the texture unaltered (Note: this part is interactive).


## CUSTOM CODE DIRECTORIES

### STYLE TRANSFER (custom/styletransfer)

	•	Code is loaded by AiPainter.ipynb.
	•	styeltransfer.py: contains all network code for the style transfer process including loss adjustment though mask, Sl1 loss functions, average pooling with LBFGS optimiser and convergence tests. 
	•	utils.py: contains all image processing utilities such as mask extraction, style tiling, noise mixing, input resolution control, input normalisation, colour swapping and quartering for upscaling. 

### IMAGE ANALYSIS (custom/imageanalysis)

	•	Runnable code.
	•	ImageAnalysis.ipynb: program to perform image analysis on 5000 paintings for each of the five styles. 
	•	ImageAnalysis.np files: numpy files generated by ImageAnalysis.ipynb and used by AiPainter.ipynb to filter the paintings.
	•	Prototype.ipynb: early prototype for image generation/analysis/filtering/interpolation using open source low resolution styleGAN models.
	•	haarcascade_frontalface_default.xml: provided by Dlib to perform face detection on paintings.

### TRAINING (custom/training)

	•	Runnable code.
	•	Training.ipynb programs: training scripts for all models created for project. Automatically downloads datasets and latest models from my google drive. 
	•	Data directory: area where models are saved if the training scripts are run (currently no models are stored due to large sizes 364mb). Contains metrics and associated logs for all models referenced by report.
	•	Please note: The larger datasets > 10k images (All_part1, All_part2, Impr_style) require Colab Pro subscription for extra disk space.

### DATA PREP (custom/dataprep)

	•	Code for inspection only. To run, database and json files need to be downloaded from https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset (large size approx 26gb).
	•	wikiart_crop_analysis.py: applies face detection to whole wikiart database and saves out crop information to centre on faces.
	•	wikiart_image_analysis.py: applies image analysis to whole wikiart database and saves out image characteristics.
	•	DB_wikiart.py programs: creates new databases based on image analysis and style/genre information using downloaded database class files.

## IMAGE DIRECTORIES

	•	content/: input images for style transfer.
	•	results/: output paintings.
	•	plots/: output plots when turned on in utils.py.
	•	style/: style image for style transfer tests.
	•	animation/: example stylegan interpolation video.

## LIBRARY CODE

### STYLEGAN2 (lib/stylegan2)

	•	Code loaded by AiPainter.ipynb for training and generating images with StyleGAN.
	•	Code licensed by Nvidia.
	•	Code has small modifications by Skyflynil for Colab training.

### ESRGAN (lib/esrgan)

	•	Code loaded by AiPainter.ipynb for upscaling with ESRGAN architecture.
	•	Code licensed by Apache.

## DEV CODE

### STYLE TRANSFER TESTS (dev/styletransfer_comparisons)
	•	Runnable code.
	•	Automatically downloads models and clones github open source code.
	•	Four models for Style Transfer comparison of results.
	•	Includes original INM705 coursework code.
