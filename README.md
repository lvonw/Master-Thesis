# Master Thesis

Terrain Generation using diffusion models

Leopold von Wendt


## Datasets

For a closer inspection of each dataset the free and open source program qgis
is highly recommended.

| Field                  | Dataset  | Source                                         |
| ---------------------- | -------  | ---------------------------------------------- |
| Topography             | SRTM     | NASA, SRTM GL1                                 |
| Geology                | GLiM     | Hartmann Moosdorf 2012 Coarse Raster           |
| Climate                | Climate  | Peel et al. 2007 Koeppen-Geiger Model          |
| Soil                   | DSMW     | Land and Water Development Division, FAO, Rome |
| Terrain Classification | GTC      | Iwahashi et al.                                |


## Sources

Guidance on implementing Diffusion Models:
- https://www.youtube.com/watch?v=ZBKpAp_6TGI
- https://www.youtube.com/watch?v=TBCRlnwJtZU
- https://github.com/Stability-AI/stablediffusion


## Dependencies

Should installing the dependancies via the provided `environment.yaml` not work
you can install the following packages manually:

- Python 3.11
- Pytorch 2.4.1+
- GDAL
- MatPlotLib
- tqdm
- pyyaml
