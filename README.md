# Electric Vehicle Charging Simulation

To run the simulation make sure the AESO_2020_demand_price.csv file is in the same directory as the simulation.py file.

To run use:

```
python simulation.py --n --id_run --pen --scale
```

- n:      number of iterations, default 10 (int)\
- id_run: file name to save results, default 'test' (str)\
- pen:    the market penetration of EVs, in number of EVs, default 0.1 (float)\
- scale:  the scaling factor for the model, default 1000 (int)

stats_output_with_v2g.py and stats_output_no_v2g.py are used to generate statistics after the model has been trained.

Requrements:

- numpy
- pandas
- tqdm
- argprase
