# Design and Implementation of a Real-Time Brain–Computer Interface for Predictive EEG-Based Communication Using Machine Learning and Natural Language Processing 

## ENGINEERING ASPECT / RESEARCH ASPECT:
The thesis focuses on the engineering design and implementation of a real-time data processing pipeline capable of acquiring, filtering, and classifying steady-state visual evoked potentials (SSVEP) from raw EEG streams. The research aspect involves evaluating different spatial filtering techniques (e.g., TRCA, CCA) and machine learning algorithms, combined with an NLP-based predictive text engine, to maximize the Information Transfer Rate (ITR) and classification accuracy.

## PURPOSE OF THE THESIS:
The main purpose of this thesis is to develop a fully functional, non-invasive BCI software system that allows users with severe motor dysfunctions to communicate efficiently. The system aims to significantly reduce user fatigue and improve communication speed by integrating intelligent word prediction (NLP) with real-time bioelectric signal classification.

## TASKS TO BE PERFORMED / SCOPE OF THE THESIS:
1. Review of current literature on BCI systems, SSVEP paradigms, and real-time EEG signal processing methods.
2. Design and implementation of a graphical user interface with precise stimulation timing.
3. Implementation of an API for real-time hardware integration and EEG data acquisition.
4. Development of a signal processing pipeline, including artifact removal and temporal/spatial filtering.
5. Training and evaluation of machine learning / statistical classification algorithms (e.g., CCA, TRCA) using offline data.
6. Integration of a Natural Language Processing (NLP) model for real-time predictive text generation to enhance the Information Transfer Rate.
7. System integration, conducting online experiments, and evaluating the system's overall performance (Accuracy, ITR).

## My equipment
brain access 32 channel eeg cap, screen with 60hz and 120hz refresh rate. 

## BIBLIOGRAPHY:
[1] Chen, X., Wang, Y., Nakanishi, M., Gao, X., Jung, T.-P., & Gao, S. "High-speed spelling with a noninvasive brain–computer interface". Proceedings of the National Academy of Sciences, 112(44), E6058-E6067. https://doi.org/10.1073/pnas.1508080112 (2015)
[2] Chen, X., Chen, Z., Gao, S., & Gao, X. "A high-ITR SSVEP-based BCI speller". Brain-Computer Interfaces, 1(3-4), 181-191. https://www.researchgate.net/publication/269998531_A_high-ITR_SSVEP-based_BCI_speller (2014)
[3] Liu, B., Huang, X., Wang, Y., Chen, X., & Gao, X. "BETA: A Large Benchmark Database Toward SSVEP-BCI Application". Frontiers in Neuroscience, 14, 627. https://doi.org/10.3389/fnins.2020.00627 (2020)
[4] Nakanishi, M., Wang, Y., Chen, X., Wang, Y.-T., Gao, X., & Jung, T.-P. "Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis". IEEE Transactions on Biomedical Engineering, 65(1), 104-112. https://doi.org/10.1109/TBME.2017.2694818 (2018)