# Information Retrieval using BERT-rankers 
We investigate BERT-based rankers performance on the Cranfield collection, which comes with full relevance judgment on all documents in the collection. The codebase based on [Bert-meets-Cranfiled](https://gitlab.science.ru.nl/nghasemi/bert-meets-cranfield) repo.
# Requirements
Make sure to install all the neccessary packages using the following command:
```bash
pip install -r requirements.txt
```

# Data Preparation
Cranfield collection was downloaded from [here](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/).

We converted the data into `json` format. You can find the sample for the queries file and the corpus file.
- The queries file
```json
[
    {
        "query_id": 0,
        "query": "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft"
    },
    {
        "query_id": 1,
        "query": "what are the structural and aeroelastic problems associated with flight of high speed aircraft"
    },
    {
        "query_id": 2,
        "query": "what problems of heat conduction in composite slabs have been solved so far"
    },
    {
        "query_id": 3,
        "query": "can a criterion be developed to show empirically the validity of flow solutions for chemically reacting gas mixtures based on the simplifying assumption of instantaneous local chemical equilibrium"
    },
    ...
```
- The corpus file
```json
[
    {
        "sentence_id": 0,
        "sentences": "experimental investigation of the aerodynamics of a wing in a slipstream an experimental study of a wing in a propeller slipstream was made in order to determine the spanwise distribution of the lift increase due to slipstream at different angles of attack of the wing and at different free stream to slipstream velocity ratios the results were intended in part as an evaluation basis for different theoretical treatments of this problem the comparative span loading curves, together with supporting evidence, showed that a substantial part of the lift increment produced by the slipstream was due to a /destalling/ or boundary-layer-control effect the integrated remaining lift increment, after subtracting this destalling lift, was found to agree well with a potential flow theory an empirical evaluation of the destalling effects was made for the specific configuration of the experiment"
    },
    {
        "sentence_id": 1,
        "sentences": "simple shear flow past a flat plate in an incompressible fluid of small viscosity in the study of high-speed viscous flow past a two-dimensional body it is usually necessary to consider a curved shock wave emitting from the nose or leading edge of the body consequently, there exists an inviscid rotational flow region between the shock wave and the boundary layer such a situation arises, for instance, in the study of the hypersonic viscous flow past a flat plate the situation is somewhat different from prandtl's classical boundary-layer problem in prandtl's original problem the inviscid free stream outside the boundary layer is irrotational while in a hypersonic boundary-layer problem the inviscid free stream must be considered as rotational the possible effects of vorticity have been recently discussed by ferri and libby in the present paper, the simple shear flow past a flat plate in a fluid of small viscosity is investigated it can be shown that this problem can again be treated by the boundary-layer approximation, the only novel feature being that the free stream has a constant vorticity the discussion here is restricted to two-dimensional incompressible steady flow"
    },
    {
        "sentence_id": 2,
        "sentences": "the boundary layer in simple shear flow past a flat plate the boundary-layer equations are presented for steady incompressible flow with no pressure gradient"
    },
    {
        "sentence_id": 3,
        "sentences": "approximate solutions of the incompressible laminar boundary layer equations for a plate in shear flow the two-dimensional steady boundary-layer problem for a flat plate in a shear flow of incompressible fluid is considered solutions for the boundary- layer thickness, skin friction, and the velocity distribution in the boundary layer are obtained by the karman-pohlhausen technique comparison with the boundary layer of a uniform flow has also been made to show the effect of vorticity"
    },
    {
        ...
```
## IMPORTANT NOTES:
Before you train or inference, make sure that you divide the data into k-fold. You can use the script `generate_fold.py` in `src` folder, unless you cannot run the code:
```bash
cd src/
python3 generate_fold.py <path-of-the-queries-file> <fold-data-output-folder>
```
### Example: ###
```bash
cd src/
python3 generate_fold.py ../data/cran/cran_qry.json ../data/folds_output/
```
You can also find the folds we used in this folder under `./data/folds`.

# Getting Started
You can set Hyper-parameters in the `init_args` function of `ranker.py` file. 
- Mode: Can be set to `'train'` or `'test'`
- MODEL_TYPE: We used the `amberoad/bert-multilingual-passage-reranking-msmarco` model but you can easily change it to any model listed [here](https://huggingface.co/transformers/pretrained_models.html)
- LEARNING_RATE: We tested values 2e-5, 3e-5, 5e-5 for our experiments as suggested by [BERT](https://www.aclweb.org/anthology/N19-1423/) paper.
- MAX_LENGTH: Indicates the BERT token limitation. We just tested values 128, 256 due to hardware limitation.
- BATCH_SIZE: Indicates the batch size for fine-tuning. We tested the sizes 16, 32, 64 as suggested by [BERT](https://www.aclweb.org/anthology/N19-1423/) paper.
- EPOCHS: Indicates the epoch numbers for fine-tuning. We tested 2, 3, 4 number of epochs as suggested by [BERT](https://www.aclweb.org/anthology/N19-1423/) paper.

## Training
To train the model from stratch, you can use the script `train.sh` in `./src/` folder, you also can change some parameters to define your output or the hyperparamaters as your wish.:
```bash
cd src/
bash train.sh
```
## Inference
To run the inference, make sure you prepare the data like [this](#data-preparation) before and ran the `generater_fold.py`, check the [IMPORTANT NOTES](#important-notes) for more details. 

If you've trained your model, link the model path in `infer.sh` or you can use ours in [here](https://drive.google.com/file/d/1wZpZiwvuC93tjDvBTYOsApnPZl1Kkkzc/view?usp=sharing):
```bash
cd src/
mkdir outputs_model
cd outputs_model
gdown --id 1wZpZiwvuC93tjDvBTYOsApnPZl1Kkkzc
cd ..
```
Run the command to infer with the model.

```bash
bash infer.sh
```
