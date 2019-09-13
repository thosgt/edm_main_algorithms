## DAS3H

This folder contains Python code of [_DAS3H: Modeling Student Learning and Forgetting for
Optimally Scheduling Distributed Practice of Skills_](https://arxiv.org/abs/1905.06873). Authors: [Benoît Choffin](https://github.com/BenoitChoffin), [Fabrice Popineau](https://github.com/fpopineau), Yolaine Bourda, and [Jill-Jênn Vie](https://github.com/jilljenn).

It is different from the [implementation used in the article](https://github.com/BenoitChoffin/das3h) as it is tailored for a dataset like Lalilo's one. 
It also uses the pandas library more (like the rolling function) to create the features.

### What is DAS3H ?
It is a model of student learning where there are 5 kinds of parameters to learn (same notations as in the article):
  - &alpha; : level of a student
  - &delta; : difficulty of an exercise
  - &beta; : difficulty of a knowledge component (not used in our implementation as we haven't tagged exercises with KCs yet)
  - &theta;<sub>wins, exercise, time-window </sub> (>0) : speed with which a student learns (?) a given type of exercise in a given time window
  - &theta;<sub>attempts, exercise, time-window</sub> (>0) : speed with which a student forgets (?) a given type of exercise in a given time window

Let's say we have a dataset looking like this.
#### Original dataset
| trace_id | date | student_id | exercise_id | correctness |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 1 january | 1 | 1 | 1 |
| 2 | 1 january | 1 | 1 | 0 |
| 3 | 1 january | 1 | 1 | 0 |
| 4 | 1 january | 2 | 1 | 0 |
| 5 | 1 january | 2 | 1 | 1 |
| 6 | **3 january** | 2 | 1 | 1 |

In the simplest version of the model (not using Factorisation Machines), the parameters to compute are those of a LogisticRegression on a dataset looking like this :

#### Encoded dataset
| trace_id | student_1 (&alpha;<sub>1</sub>)| student (&alpha;<sub>2</sub>)| exercise_1 (&delta;<sub>1</sub>) | exercise_2 (&delta;<sub>2</sub>)| wins_on_exo_1_in_the_past_day (&theta;<sub>wins, exo_1, one-day</sub>)| attempts_on_exo_1_in_the_past_day (&theta;<sub>attempts, exo_1, one-day</sub>) | wins_on_exo_1_in_the_past_week (&theta;<sub>wins, exo_1, one-week</sub>)| attempts_on_exo_1_in_the_past_week (&theta;<sub>attempts, exo_1, one-week</sub>)| other columns like &theta; parameters on ex 2 | 
|:-:|:-:|:-----:|:-----:|:------:|:----:|:----:|:-:|:-:|:-:|
| 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| 3 | 0 | 1 | 1 | 0 | 1 | 2 | 1 | 2 |
| 4 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 5 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| 6 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 2 |

To get this encoded dataset, we one-hot-encode on the student_id and exercise_id and add the number of previous attempts and wins that a student had in the given time windows. 

Try to write on a piece of paper what you get doing that and compare to the encoded dataset above. See the *Important* note for further details.

#### Logistic Regression

As stated in [sklearn documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), Logistic Regression is the optimization of <a href="https://www.codecogs.com/eqnedit.php?latex=\min_{w,&space;c}&space;\frac{1}{2}w^T&space;w&space;&plus;&space;C&space;\sum_{i=1}^n&space;\log(\exp(-&space;y_i&space;(X_i^T&space;w&space;&plus;&space;c))&space;&plus;&space;1)&space;." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{w,&space;c}&space;\frac{1}{2}w^T&space;w&space;&plus;&space;C&space;\sum_{i=1}^n&space;\log(\exp(-&space;y_i&space;(X_i^T&space;w&space;&plus;&space;c))&space;&plus;&space;1)&space;." title="\min_{w, c} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1) ." /></a>

Here, the parameters &alpha;, &delta;, &beta;, and &theta; are concatenated in *w*

*Important :*
As you may have noticed, the features in the *encoded* dataset seem to "lag" one trace behind the original dataset. Actually this is done to prevent any data leakage and not use the answer at time T to predict itself. If this is not clear, please tell me.

*Note :*
Actually the number of wins and number of attempts are not fed directly to the model, instead they go through a scaling function :
``` python
lambda x: log(1 + x)
```
in the article