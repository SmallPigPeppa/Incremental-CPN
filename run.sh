for lambda in 0.2 0.1 0.05 0.025 0.01 0.005 0.0025 0.001
do
    /share/wenzhuoliu/conda-envs/solo-learn/bin/python test_cpn.py --LAMBDA ${lambda}
done