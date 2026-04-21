# On-off ground check

> By logistic regression. One neural network.

## Tutorial
1. Clone this repository.
2. Use VOFA+ to import Onground and Offground data. Rename it. Finally you can get 2 csv files, named "onground.csv" and "offground.csv", replace them.
3. Run two commands in your terminal.

```bash
python3 add.py    # add tables
python3 train.py
```

4. Parameters are generated in "model_param.c". Copy them in your embedded code.
5. Notice features are normalize, so you should do it in your embedded code simultaneously. 

Example code:

```c
uint8_t ground_check(Leg_Typedef *leg, IMU_Data_t *imu, float *w, float b, float *mean, float *std)
{
    float norm[12], prob;

    norm[0] = (leg->LQR.F_0 - mean[0]) / std[0];
    norm[1] = (leg->LQR.T_p - mean[1]) / std[1];
    norm[2] = (leg->stateSpace.theta - mean[2]) / std[2];
    norm[3] = (leg->stateSpace.dtheta - mean[3]) / std[3];
    norm[4] = (leg->stateSpace.dtheta * leg->stateSpace.dtheta - mean[4]) / std[4];
    norm[5] = (leg->stateSpace.ddtheta - mean[5]) / std[5];
    norm[6] = (sin(leg->stateSpace.theta) - mean[6]) / std[6];
    norm[7] = (cos(leg->stateSpace.theta) - mean[7]) / std[7];
    norm[8] = (leg->vmc_calc.L0[POS] - mean[8]) / std[8];
    norm[9] = (leg->vmc_calc.L0[VEL] - mean[9]) / std[9];
    norm[10] = (leg->vmc_calc.L0[ACC] - mean[10]) / std[10];
    norm[11] = (imu->accel[2] - mean[11]) / std[11];

    float z = b;
    for (int i = 0; i < 12; i++)
    {
        z += w[i] * norm[i];
    }

    prob = 1.0f / (1.0f + exp(-z));


    // VOFA_justfloat(norm[0], norm[1],norm[2],norm[3],norm[4],norm[5],norm[6],norm[7],norm[8],norm[9]);
    return (prob >= 0.5) ? 1 : 0;
    
}
```