# Map OCR
 Extracts text from map images

## Introduction

Maps contain a lot of rich information such as text that might be useful to extract and search. To extract this information however is a much more difficult task than what is typically done for traditional text documents. Text in map documents face challenges like... 


## Installation

To use this library, you need PIL, numpy, opencv, colormath, and pytesseract. You also need to manually install the Tesseract text recognition engine in your environment and make sure it's on your system's environment path. 


## Basic usage

To use this library, start by importing the library and loading the map image for which you want to detect text: 

    >>> import mapocr
    >>> from PIL import Image
    >>> im = Image.open('tests/data/burkina_pol96.jpg')

![Original image](/tests/data/burkina_pol96.jpg)

Let's say we want to detect all approximately black text, that's as simple as:

    >>> textcolor = (0,0,0)
    >>> texts = mapocr.textdetect.auto_detect_text(im, textcolor)
    >>> len(texts)
    76

This results in a list of dicts defining each "word" of text along with metadata information, in the same format as what's returned by the Tesseract engine. Here are examples of the first 3 text words from our image:

    >>> for textinfo in texts[:3]:
    ...     print(textinfo)
    {'level': '5', 'page_num': '1', 'block_num': '1', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 76, 'top': 40, 'width': 104, 'height': 22, 'conf': 96.304893, 'text': 'Burkina', 'fontheight': 22, 'text_clean': 'Burkina', 'text_alphas': 'Burkina', 'color': (0, 0, 0), 'color_match': 5.196179646556982}
    {'level': '5', 'page_num': '1', 'block_num': '1', 'par_num': '1', 'line_num': '1', 'word_num': '2', 'left': 192, 'top': 40, 'width': 62, 'height': 22, 'conf': 96.304893, 'text': 'Faso', 'fontheight': 22, 'text_clean': 'Faso', 'text_alphas': 'Faso', 'color': (0, 0, 0), 'color_match': 5.704414545359642}
    {'level': '5', 'page_num': '1', 'block_num': '3', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 903, 'top': 83, 'width': 37, 'height': 10, 'conf': 91.843643, 'text': 'Menaka', 'fontheight': 10, 'text_clean': 'Menaka', 'text_alphas': 'Menaka', 'color': (0, 0, 0), 'color_match': 14.792302518904004}

In order to group these individual text words into logical text groups such as multi-word lines of texts or paragraphs, simply call the `connect_text` function:

    >>> textgroups = mapocr.textgroup.connect_text(texts)
    >>> len(textgroups)
    74

    >>> for textinfo in textgroups[:3]:
    ...     print(textinfo)
    {'text': 'Burkina Faso', 'text_clean': 'Burkina Faso', 'text_alphas': 'BurkinaFaso', 'conf': 96.304893, 'left': 76, 'top': 40, 'fontheight': 22, 'color': (0, 0, 0), 'color_match': 5.450297095958312, 'width': 178, 'height': 22}
    {'text': 'Menaka', 'text_clean': 'Menaka', 'text_alphas': 'Menaka', 'conf': 91.843643, 'left': 903, 'top': 83, 'fontheight': 10, 'color': (0, 0, 0), 'color_match': 14.792302518904004, 'width': 37, 'height': 10}
    {'text': 'dary', 'text_clean': 'dary', 'text_alphas': 'dary', 'conf': 90.281227, 'left': 250, 'top': 99, 'fontheight': 10, 'color': (0, 0, 0), 'color_match': 16.124256789325415, 'width': 24, 'height': 10}

For most cases that's pretty much all there is to it. There are however several other options, customizations, and additional features to help improve the results. 


## Excluding false positives

Not all returned text is going to be correctly parsed, and some of it may even just be graphics misinterpreted as text. Luckily, the Tesseract engine records the confidence with which it parsed each word of text, expressed as a probability from 0 to 100 percent. By default, mapocr only returns text interpreted with a probability of at least 60 percent of being correct. The users can specify a stricter or looser requirement for this confidence parameter: 

    >>> hiconf_texts = mapocr.textdetect.auto_detect_text(im, textcolor, textconf=90)
    >>> len(hiconf_texts)
    48

    >>> for textinfo in hiconf_texts[:3]:
    ...     print(textinfo)
    {'level': '5', 'page_num': '1', 'block_num': '1', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 76, 'top': 40, 'width': 104, 'height': 22, 'conf': 96.304893, 'text': 'Burkina', 'fontheight': 22, 'text_clean': 'Burkina', 'text_alphas': 'Burkina', 'color': (0, 0, 0), 'color_match': 5.196179646556982}
    {'level': '5', 'page_num': '1', 'block_num': '1', 'par_num': '1', 'line_num': '1', 'word_num': '2', 'left': 192, 'top': 40, 'width': 62, 'height': 22, 'conf': 96.304893, 'text': 'Faso', 'fontheight': 22, 'text_clean': 'Faso', 'text_alphas': 'Faso', 'color': (0, 0, 0), 'color_match': 5.704414545359642}
    {'level': '5', 'page_num': '1', 'block_num': '3', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 903, 'top': 83, 'width': 37, 'height': 10, 'conf': 91.843643, 'text': 'Menaka', 'fontheight': 10, 'text_clean': 'Menaka', 'text_alphas': 'Menaka', 'color': (0, 0, 0), 'color_match': 14.792302518904004}


## Faint or blurry text

In order to differentiate the pixels that make up text of a particular color from the pixels that make up background pixels of various other colors, mapocr uses a novel color thresholding technique called Delta E 2000 which is based on how humans perceive color differences. Mapocr by default isolates all pixels that have a Delta E value less than 25, which accurately finds the relevant color while allowing for some variation. A Delta E of 10 is generally the smallest possible difference between two colors that a human can perceive. 

In maps where the text is particularly faint, pixelated, or blended with the background, the text color may span a wider range of colors and is therefore more difficult to pick up. In these cases, the user can specify a higher threshold value for the Delta E parameter, which should return a longer - but potentially more noisy - list of potential text: 

    >>> faint_texts = mapocr.textdetect.auto_detect_text(im, textcolor, colorthresh=50)
    >>> len(faint_texts)
    76


## Multiple text colors

If there's multiple types of text with different colors, we can also do that. In our example image, let's say we want to pick up all the black text as before, but also all the dark green text used to represent the names of administrative divisions. To do so we simply pass a list of colors to the textcolor arg: 

    >>> black = (0,0,0)
    >>> green = (80,104,64)
    >>> textcolors = [black, green]
    >>> texts = mapocr.textdetect.auto_detect_text(im, textcolors)

Each text dict contains an entry indicating which of the input colors that text is most similar to, and we can use that to filter text by color. First, the black text:

    >>> blacktexts = [textinfo for textinfo in texts
    ...                 if textinfo['color'] == black]
    >>> len(blacktexts)
    76

    >>> for textinfo in blacktexts[:3]:
    ...     print(textinfo)
    {'level': '5', 'page_num': '1', 'block_num': '1', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 76, 'top': 40, 'width': 104, 'height': 22, 'conf': 96.304893, 'text': 'Burkina', 'fontheight': 22, 'text_clean': 'Burkina', 'text_alphas': 'Burkina', 'color': (0, 0, 0), 'color_match': 5.196179646556982}
    {'level': '5', 'page_num': '1', 'block_num': '1', 'par_num': '1', 'line_num': '1', 'word_num': '2', 'left': 192, 'top': 40, 'width': 62, 'height': 22, 'conf': 96.304893, 'text': 'Faso', 'fontheight': 22, 'text_clean': 'Faso', 'text_alphas': 'Faso', 'color': (0, 0, 0), 'color_match': 5.704414545359642}
    {'level': '5', 'page_num': '1', 'block_num': '3', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 903, 'top': 83, 'width': 37, 'height': 10, 'conf': 91.843643, 'text': 'Menaka', 'fontheight': 10, 'text_clean': 'Menaka', 'text_alphas': 'Menaka', 'color': (0, 0, 0), 'color_match': 14.792302518904004}

And then the green text:

    >>> greentexts = [textinfo for textinfo in texts
    ...                 if textinfo['color'] == green]
    >>> len(greentexts)
    52

    >>> for textinfo in greentexts[:3]:
    ...     print(textinfo)
    {'level': '5', 'page_num': '1', 'block_num': '4', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 76, 'top': 40, 'width': 104, 'height': 23, 'conf': 90.983246, 'text': 'Burkina', 'fontheight': 23, 'text_clean': 'Burkina', 'text_alphas': 'Burkina', 'color': (80, 104, 64), 'color_match': 18.014517214584213}
    {'level': '5', 'page_num': '1', 'block_num': '4', 'par_num': '1', 'line_num': '1', 'word_num': '2', 'left': 192, 'top': 40, 'width': 64, 'height': 23, 'conf': 94.53508, 'text': 'Faso', 'fontheight': 23, 'text_clean': 'Faso', 'text_alphas': 'Faso', 'color': (80, 104, 64), 'color_match': 18.290441354618572}
    {'level': '5', 'page_num': '1', 'block_num': '5', 'par_num': '1', 'line_num': '1', 'word_num': '1', 'left': 640, 'top': 50, 'width': 18, 'height': 10, 'conf': 72.97908, 'text': 'Gea', 'fontheight': 10, 'text_clean': 'Gea', 'text_alphas': 'Gea', 'color': (80, 104, 64), 'color_match': 17.987220345780386}



