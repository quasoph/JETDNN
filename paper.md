---
title: 'JETDNN: a package for developing plasma pedestal models with Deep Learning.'
tags:
  - Python
  - plasma physics
  - tokamak science
  - deep learning
  - neural networks
authors:
  - name: Sophie Frankel
    orcid: 0009-0008-9807-7175
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: James Simpson
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: E. Solano
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "2, 3"
affiliations:
 - name: Newcastle University, Newcastle Upon Tyne, UK
   index: 1
 - name: Culham Centre for Fusion Energy, Abingdon, Oxfordshire, UK
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 07 September 2023
bibliography: paper.bib

---

# Summary

Modelling plasma edge pedestals is a crucial step in the path to self-sustaining nuclear fusion reactions. In this paper we introduce JETDNN, a Python framework for building analytic models relating plasma pedestal heights with multiple engineering parameters, using Deep Neural Networks (DNNs). JETDNN makes modelling straightforward using a 3-part workflow: data inspection, model building, and visualisation. Users can find relevant engineering parameters with jetdnn.interpret, build predictive models with jetdnn.predict, and visualise both models and predictions in multiple ways with jetdnn.visualise. JETDNN offers functionality where other ML pedestal predictive software does not, in that trained models used to predict pedestal height values can also be represented analytically.

# Statement of need

In plasma physics, there is a strong need to produce models relating engineering parameters to H-mode edge pedestals in order to mitigate instabilities called Edge Localised Modes (ELMs). ELMs prevent current tokamak fusion reactions from becoming self-sustaining by emitting energy and damaging the surfaces of the plasma containment vessel. Models of edge pedestals can give information on the load exerted by ELMs on the divertor (T. Hatae et al.), so that changes can be made to mitigate these instabilities. Multi-input deep neural networks shine in areas like tokamak science where many characteristics affect a single output, in which analytic models may not be able to find links between 5+ characteristics. `JETDNN` employs a nonlinear neural network model, which is most effective for pedestal predictions in DIII-D data (E. U Zeger et al. 2021), however no studies to date have been performed on JET laboratory pedestal data. `JETDNN` aims to allow users to develop accurate machine learning models to find relationships in JET pedestal data.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from XXX.

# References