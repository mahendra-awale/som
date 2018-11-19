/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package kohnenMap;

import java.awt.Button;
import java.awt.Color;
import java.awt.GridLayout;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;

/**
 *
 * @author mahendra
 */
public class NewJFrame extends javax.swing.JFrame {

    /**
     * Input parameters for KOHNEN map
     */
    //input data for map
    double input_data[][] = new double[][]{{0., 0., 0.}, {0., 0., 1.}, {0., 0., 0.5},
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

    //kohnonen map training object
    trainKohnenMap tkm;

    //map size
    int size = 50;

    //learning rate
    double lr = 1;

    //decay_rate
    double dr = 0.01;

    //iterations
    int iter = 100;

    /**
     * Creates new form NewJFrame
     */
    public NewJFrame() {

        initComponents();

        tkm = new trainKohnenMap(size, input_data, lr, dr);

        jPanel1.setLayout(new GridLayout(size, size));

        for (int a = 0; a < tkm.kmap.length; a++) {
            for (int b = 0; b < tkm.kmap[a].length; b++) {

                Color clr = new Color((float) tkm.kmap[a][b][0], (float) tkm.kmap[a][b][1], (float) tkm.kmap[a][b][2]);
                Button jbn1 = new Button("");
                jbn1.setBackground(clr);
                jPanel1.add(jbn1);
            }
        }
    }

    //==========================================================================
    /**
     * Updating the map display
     *
     * @param kmap
     */
    public void updateMapDisplay(double kmap[][][]) {

        int counter = 0;
        for (int a = 0; a < kmap.length; a++) {
            for (int b = 0; b < kmap[a].length; b++) {

                Color clr = new Color((float) kmap[a][b][0], (float) kmap[a][b][1], (float) kmap[a][b][2]);
                jPanel1.getComponent(counter).setBackground(clr);
                counter++;
            }
        }
    }

    //==========================================================================
    /**
     * Updating the map display
     * @param kmap
     */
    public void resizeMap(double kmap[][][]) {

        jPanel1.removeAll();
        jPanel1.setLayout(new GridLayout(size, size));
        

        for (int a = 0; a < kmap.length; a++) {
            for (int b = 0; b < kmap[a].length; b++) {

                Color clr = new Color((float) kmap[a][b][0], (float) kmap[a][b][1], (float) kmap[a][b][2]);
                Button jbn1 = new Button("");
                jbn1.setBackground(clr);
                jPanel1.add(jbn1);
            }
        }
        
        jPanel1.revalidate();
        jPanel1.repaint();
        this.revalidate();
        this.repaint();
    }
    //==========================================================================

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        jPanel2 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        mapsizeIn = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        learning_rateIn = new javax.swing.JTextField();
        jLabel3 = new javax.swing.JLabel();
        decay_rateIn = new javax.swing.JTextField();
        jButton1 = new javax.swing.JButton();
        jLabel4 = new javax.swing.JLabel();
        iterIn = new javax.swing.JTextField();
        resizemap = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel1.setBorder(new javax.swing.border.MatteBorder(null));
        jPanel1.setLayout(new java.awt.GridLayout());

        jPanel2.setBackground(java.awt.SystemColor.controlShadow);
        jPanel2.setBorder(new javax.swing.border.MatteBorder(null));

        jLabel1.setText("Size of Map");

        mapsizeIn.setText("50");
        mapsizeIn.addInputMethodListener(new java.awt.event.InputMethodListener() {
            public void inputMethodTextChanged(java.awt.event.InputMethodEvent evt) {
                mapsizeInInputMethodTextChanged(evt);
            }
            public void caretPositionChanged(java.awt.event.InputMethodEvent evt) {
                mapsizeInCaretPositionChanged(evt);
            }
        });
        mapsizeIn.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                mapsizeInActionPerformed(evt);
            }
        });
        mapsizeIn.addPropertyChangeListener(new java.beans.PropertyChangeListener() {
            public void propertyChange(java.beans.PropertyChangeEvent evt) {
                mapsizeInPropertyChange(evt);
            }
        });

        jLabel2.setText("Learning rate");

        learning_rateIn.setText("0.5");
        learning_rateIn.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                learning_rateInActionPerformed(evt);
            }
        });

        jLabel3.setText("Decay rate");

        decay_rateIn.setText("0.01");

        jButton1.setText("RUN");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jLabel4.setText("Iterations");

        iterIn.setText("100");

        resizemap.setText("resize");
        resizemap.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                resizemapActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(22, 22, 22)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel2)
                            .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                .addComponent(iterIn, javax.swing.GroupLayout.Alignment.LEADING)
                                .addComponent(jLabel4, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(jButton1, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(decay_rateIn, javax.swing.GroupLayout.Alignment.LEADING)
                                .addComponent(jLabel3, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, 80, Short.MAX_VALUE)
                                .addComponent(learning_rateIn, javax.swing.GroupLayout.Alignment.LEADING)))
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                            .addComponent(mapsizeIn, javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(resizemap, javax.swing.GroupLayout.DEFAULT_SIZE, 65, Short.MAX_VALUE)
                        .addGap(6, 6, 6))))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(mapsizeIn, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(resizemap))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(learning_rateIn, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel3)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(decay_rateIn, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel4)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(iterIn, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(jButton1)
                .addContainerGap(14, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, 341, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    //==========================================================================
    private void mapsizeInActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_mapsizeInActionPerformed
        // TODO add your handling code here:

    }//GEN-LAST:event_mapsizeInActionPerformed

    private void learning_rateInActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_learning_rateInActionPerformed
        // TODO add your handling code here:     
    }//GEN-LAST:event_learning_rateInActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:


        try {
            lr = Double.valueOf(this.learning_rateIn.getText());
        } catch (Exception e) {
            return;
        }

        try {
            dr = Double.valueOf(this.decay_rateIn.getText());
        } catch (Exception e) {
            return;
        }

        try {
            iter = Integer.valueOf(this.iterIn.getText());
        } catch (Exception e) {
            return;
        }

        tkm.decay_rate = dr;
        tkm.learning_rate = lr;
        tkm.run(iter, this);
    }//GEN-LAST:event_jButton1ActionPerformed

    private void mapsizeInInputMethodTextChanged(java.awt.event.InputMethodEvent evt) {//GEN-FIRST:event_mapsizeInInputMethodTextChanged
    }//GEN-LAST:event_mapsizeInInputMethodTextChanged

    private void mapsizeInCaretPositionChanged(java.awt.event.InputMethodEvent evt) {//GEN-FIRST:event_mapsizeInCaretPositionChanged
    }//GEN-LAST:event_mapsizeInCaretPositionChanged

    private void mapsizeInPropertyChange(java.beans.PropertyChangeEvent evt) {//GEN-FIRST:event_mapsizeInPropertyChange
        // TODO add your handling code here:
    }//GEN-LAST:event_mapsizeInPropertyChange

    private void resizemapActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_resizemapActionPerformed
        // TODO add your handling code here:
                //check for parameters
        try {
            size = Integer.valueOf(this.mapsizeIn.getText());
            tkm = new trainKohnenMap(size, input_data, lr, dr);
            resizeMap(tkm.kmap);
        } catch (Exception e) {
            return;
        }
    }//GEN-LAST:event_resizemapActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(NewJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(NewJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(NewJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(NewJFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new NewJFrame().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField decay_rateIn;
    private javax.swing.JTextField iterIn;
    private javax.swing.JButton jButton1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JTextField learning_rateIn;
    private javax.swing.JTextField mapsizeIn;
    private javax.swing.JButton resizemap;
    // End of variables declaration//GEN-END:variables
}
