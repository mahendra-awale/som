/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kohnenMap;

import java.util.ArrayList;

/**
 * @author mahendra
 */
public class trainKohnenMap {

    //x-y: position of a neuron on a grid
    //z: wts accosiated with the a neuron
    //size of x and y determine by size of map
    //size of z is equal to inpput vector size
    double kmap[][][];

    //learning rate
    double learning_rate;

    //decay rate
    double decay_rate;

    //sigma for gaussian wts;
    double sigma;

    //input data
    double input_data[][];

    //initial radius for nearest neighbours
    double nn_r;

    //Constrcutor
    trainKohnenMap(int size, double inputdata[][], double lr, double dr) {

        //Initialized the map
        kmap = new double[size][size][inputdata[0].length];

        //learnhing rate
        learning_rate = lr;

        //decay rate
        decay_rate = dr;

        //input data
        input_data = inputdata;

        //nearest neighbours radius
        nn_r = size / 2.0;

        //init wts
        kmap = wts_init(kmap);
    }

    void run(int num_iter) {

        //initi
        for (int t = 0; t < num_iter; t++) {
            for (int a = 0; a < input_data.length; a++) {

                //get bmu
                double[] bmu = get_bestMatchingNeuron(kmap, input_data[a]);

                //get nearest neighbours of bmu
                ArrayList<double[]> neighbours = get_Neighbours(bmu, kmap, nn_r, decay_rate, t);

                //update the map
                update_wts(kmap, neighbours, bmu, input_data[a], learning_rate, decay_rate, t);
            }

            System.out.println("DONE WITH ITER " + t);
        }
    }

    void run(int num_iter, NewJFrame frame) {

        //initi
        for (int t = 0; t < num_iter; t++) {
            for (int a = 0; a < input_data.length; a++) {

                //get bmu
                double[] bmu = get_bestMatchingNeuron(kmap, input_data[a]);

                //get nearest neighbours of bmu
                ArrayList<double[]> neighbours = get_Neighbours(bmu, kmap, nn_r, decay_rate, t);

                //update the map
                update_wts(kmap, neighbours, bmu, input_data[a], learning_rate, decay_rate, t);

            }

            frame.updateMapDisplay(kmap);

            System.out.println("DONE WITH ITER " + t);
        }
    }

    //==========================================================================
    //initialized the wts on map
    double[][][] wts_init(double map[][][]) {

        for (int a = 0; a < map.length; a++) {
            for (int b = 0; b < map[a].length; b++) {
                for (int c = 0; c < map[a][b].length; c++) {
                    map[a][b][c] = Math.random();
                }
            }
        }

        return map;
    }
    //==========================================================================

    //calculate euclidean distance
    double euclidean_distance(double vec1[], double vec2[]) {

        double dist = 0;
        for (int a = 0; a < vec1.length; a++) {
            dist = dist + ((vec1[a] - vec2[a]) * (vec1[a] - vec2[a]));
        }

        dist = Math.sqrt(dist);
        return dist;
    }

    //==========================================================================
    //get best matching unit or neuron position (x and y).
    double[] get_bestMatchingNeuron(double map[][][], double input[]) {

        double winnerDistance = Double.MAX_VALUE;
        double winnerNeuron[] = new double[2];

        for (int a = 0; a < map.length; a++) {

            for (int b = 0; b < map.length; b++) {

                double ecd = euclidean_distance(map[a][b], input);
                if (ecd < winnerDistance) {
                    winnerDistance = ecd;
                    winnerNeuron[0] = a;
                    winnerNeuron[1] = b;
                }
            }
        }

        return winnerNeuron;
    }

    //==========================================================================
    //get neighbours of best matching neuron
    /**
     *
     * @param bmu: x-y position of best matching neuron
     * @param map: kohnen map (3d array)
     * @param initr: initial radius we started with
     * @param dr: decay constant
     * @param t: time step (it's iteration number)
     *
     * using exponential decay formula: a*(1-k)^t
     *
     */
    ArrayList<double[]> get_Neighbours(double bmu[], double map[][][], double initr, double dr, double t) {

        double current_radius = initr * Math.pow((1 - dr), t);
        ArrayList<double[]> nn = new ArrayList();
        for (int i = 0; i < map.length; i++) {
            for (int j = 0; j < map[i].length; j++) {
                double ecd = euclidean_distance(bmu, new double[]{i, j});
                if (ecd <= current_radius) {
                    nn.add(new double[]{i, j});
                }
            }
        }

        return nn;
    }

    //==========================================================================
    //updating the wts of neurons in a map
    /**
     *
     * @param map
     * @param nnIndexes
     * @param bmuIndx
     * @param input
     * @param lr
     * @param dr
     * @param t
     * @return
     */
    double[][][] update_wts(double map[][][], ArrayList<double[]> nnIndexes, double[] bmuIndx, double input[], double lr, double dr, double t) {

        //Current learning rate
        double c_lr = lr * Math.pow((1 - dr), t);

        //Update the wts
        for (int a = 0; a < nnIndexes.size(); a++) {

            //position of a nearent neuron of bmu
            int neuron_xpos = (int) nnIndexes.get(a)[0];
            int neuron_ypos = (int) nnIndexes.get(a)[1];

            //calculate position distance between this neuron and bmu
            double ecd = euclidean_distance(nnIndexes.get(a), bmuIndx);

            //calculate gauss wts for this neuron
            //System.out.println(ecd+" "+ecd*0.20);
            double gausswt = gaussian_wts(0, ecd, 3);

            //updating wts of a neuron
            for (int b = 0; b < map[neuron_xpos][neuron_ypos].length; b++) {
                map[neuron_xpos][neuron_ypos][b] = map[neuron_xpos][neuron_ypos][b] + (gausswt * (c_lr * (input[b] - map[neuron_xpos][neuron_ypos][b])));
            }
        }

        return map;
    }

    //==========================================================================
    static double gaussian_wts(double u, double x, double sigma) {
        double x1 = Math.exp(-(1.0 / 2.0) * Math.pow((x - u) / sigma, 2));
        return x1;
    }

    //==========================================================================
    public static void main(String args[]) {

//        double input[][] = new double[100][3];
//        for (int a = 0; a < 100; a++) {
//            input[a][0] = Math.random();
//            input[a][1] = Math.random();
//            input[a][2] = Math.random();
//        }
        double input[][] = new double[][]{{0., 0., 0.}, {0., 0., 1.}, {0., 0., 0.5},
        {0.125, 0.529, 1.0},
        {0.33, 0.4, 0.67},
        {0.6, 0.5, 1.0},
        {0., 1., 0.},
        {1., 0., 0.},
        {0., 1., 1.},
        {1., 0., 1.},
        {1., 1., 0.},
        {1., 1., 1.},
        {.33, .33, .33},
        {.5, .5, .5},
        {.66, .66, .66}};

        trainKohnenMap ok = new trainKohnenMap(50, input, 1, 0.01);

        for (int a = 0; a < ok.kmap.length; a++) {
            for (int b = 0; b < ok.kmap[a].length; b++) {
                System.out.println(a + ";" + b + " " + ok.kmap[a][b][0] + ";" + ok.kmap[a][b][1] + ";" + ok.kmap[a][b][2]);
            }
        }

        System.out.println("============================");
        ok.run(500);

        for (int a = 0; a < ok.kmap.length; a++) {
            for (int b = 0; b < ok.kmap[a].length; b++) {
                System.out.println(a + ";" + b + " " + ok.kmap[a][b][0] + ";" + ok.kmap[a][b][1] + ";" + ok.kmap[a][b][2]);

            }
        }
    }
}
