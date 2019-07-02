Computational Model
===================

This part of the documentation specifies the functional forms, types of shocks and exogenous processes we bring together with the developed economic model to perform simulation and estimation. We discuss the functional form of the utility function, the laws of motion for the agents' budget constraints, the chosen formulation of the wage and the experience accumulation processes, etc. in greater detail.

Toy model
*********

This section lays out the computational model corresponding to the current version of the code.


Structural equations
---------------------

In the toy model, the individual's utility is given by:

.. math::

	u(c_t, l_t; \theta) = \frac{c_t^\mu}{\mu}exp
	\begin{cases}
	0, & if \hspace{4pt} l_t = N,
	\\
	\theta_P, & if \hspace{4pt} l_t = P,
	\\
	\theta_F, & if \hspace{4pt} l_t = F,
	\end{cases}

Unobserved heterogeneity in the model is reflected in the fact that the choice specific parameters for the disutility
of work can take on multiple values thus reflecting the individual's type.
In assuming this form of utility, the toy model abstracts away from some components we would like to include
in the enhanced model. First, the above utility function does not include any covariates and corresponding coefficients
in :math:`U(.)` apart from the choice specific constant :math:`\theta_l`. Second, it abstracts from weighting
consumption by an equivalence scale.

The budget constraint in the toy model is given by :math:`c_t = h_t w_t`.

The wage equation is given by:

.. math::

	\begin{split}ln \hspace{2pt} w_t^m & = \gamma_{s,0}  + \gamma_{s,1} ln(e_t + 1) + \xi_t,\\
	ln \hspace{2pt} w_t & = ln \hspace{2pt} w_t^m - \xi_t,\\
	e_t & = (e_P*g_{sP} + e_F)(1-\delta_s),\\\end{split}


where :math:`e_P` and :math:`e_F` measure the total years in part-time and full-time experience accumulated up to period :math:`t`.


Transformatons
--------------

To make the calculation of the flow utilities and value functions more explicit we substitute the above equation
in one another. As a result we arrive at a single term that represents :math:`u(c_t, l_t, \theta)`.

First we substitute the experience term in the first line of the wage equation,

.. math::
	
	ln \hspace{2pt} w_t^m = \gamma_{s,0}  + \gamma_{s,1} ln[(e_P \hspace{2pt} \cdot \hspace{2pt} g_{sP} + e_F)(1-\delta_s) + 1] + \xi_t,

and take the exponent of both sides of the equation:

.. math::
	
	w_t^m = exp \hspace{2pt} {\gamma_{s,0}} \hspace{2pt} \cdot \hspace{2pt} exp\{\gamma_{s,1} \hspace{2pt} \cdot \hspace{2pt} ln[(e_P*g_{sP} + e_F)(1-\delta_s) + 1]\} \hspace{2pt} \cdot \hspace{2pt} exp \hspace{2pt} {\xi_t},

We then substitute the wage in the budget constraint:

.. math::
	
	c_t = h_t \hspace{2pt} \cdot \hspace{2pt} \{exp \hspace{2pt} {\gamma_{s,0}} \hspace{2pt} \cdot \hspace{2pt} exp\{\gamma_{s,1} \hspace{2pt} \cdot \hspace{2pt} ln[(e_P*g_{sP} + e_F)(1-\delta_s) + 1]\} \hspace{2pt} \cdot \hspace{2pt} exp \hspace{2pt} {\xi_t}\},

And we arrive at the final expression by substituting consumption in the utility function:

.. math::
	
	\begin{split}
	u(c_t, l_t; \theta) & = \frac{h_t^\mu \hspace{2pt} \cdot \hspace{2pt} \{exp \hspace{2pt} {\gamma_{s,0}} \hspace{2pt} \cdot \hspace{2pt} exp\{\gamma_{s,1} \hspace{2pt} \cdot \hspace{2pt} ln[(e_P*g_{sP} + e_F)(1-\delta_s) + 1]\} \hspace{2pt} \cdot \hspace{2pt} exp \hspace{2pt} {\xi_t}\}^\mu } \hspace{2pt} \\
	& \cdot \hspace{2pt} \hspace{2pt} exp
	\begin{cases}
	0, & if \hspace{4pt} l_t = N,
	\\
	\theta_P, & if \hspace{4pt} l_t = P,
	\\
	\theta_F, & if \hspace{4pt} l_t = F,
	\end{cases}\end{split}\\


Finally, the distribution of the error term is assumed to be multivariate normal with zero means.


Enhanced model
**************

The goal of development efforts is directed to gradually extending the toy model towards the enhanced model presented below.

Outline of model components
----------------------------

In this framework women are modelled between the age when they start working after having completed
education and 60 years of age. They retire at the age of 60 and live for additional 10 years using their
accumulated savings. No re-entry in education is possible in the model. Having completed education,
in each period (year) of their life, women make consumption and labor supply choices.
They choose between nonemployment (N), part-time (P), or full-time employment (F).
Female workers face labor-market frictions.

In the first period of observation, each woman draws a random preference for work, consisting of a utility cost
of part-time work (:math:`\theta_P`) and a utility-cost of full-time work (:math:`\theta_F`).
The utility cost parameters,  :math:`\theta_F` and :math:`\theta_P`, can take on two (or more) values each,
i.e., there are, e.g., two types: high - type I, and low - type II. The values of the baseline type coefficients are
normalised to zero. Both parameter values, :math:`\theta_F` and :math:`\theta_P` for type I, as well as the
frequency of type I individuals in the data are estimated alongside the other free parameters of the model.

The tax and welfare system is year specific reflecting actual real world changes.
It defines disposable income for each employment option. Households are credit constrained, i.e., they cannot borrow.
Finally, the following elements enter the computational model as exogenous processes:
childbirth, marriage, divorce and the male wage process.

Structural equations
---------------------

Instantaneous utility
^^^^^^^^^^^^^^^^^^^^^^

The individuals' flow utility is given by:

.. math::

	u(c_t, l_t; \theta, Z_t) = \frac{(c_t/n_t)^\mu}{\mu}exp\{U(l_t, \theta, Z_t)\}

	U(l_t, \theta, Z_t) =
	\begin{cases}
	0, & \text{if $l_t = N$,}
	\\[4pt]
	\theta_l + Z'_t\alpha(l_t), & \text{if $l_t = P$ or $F$},
	\end{cases}

where :math:`\alpha(l_t) = \alpha_F + \alpha_P \cdot \bf{1}` :math:`(l_t = P)`.

The CRRA utility depends on consumption per adult equivalent, female labor supply :math:`l`, characteristics :math:`Z`, and {\diw preference for work :math:`\theta`}. :math:`Z` can contain information on marital status, presence of children, their interaction, dummies for children in different age groups, an indicator whether or not the partner is working, etc. :math:`U(.)` of not working is normalised to zero; :math:`\beta` is set to 0.98;

There are several implications of the choice of this particular form of the utility function.
Given the above form, instantaneous utility is non-separable in consumption and leisure.
Total (lifetime) utility is the sum of CRRA functions, i.e., it is additively separable intertemporaneously.
:math:`\mu` is the curvature parameter that governs risk-aversion and the elasticity of intertemporal substitution.
The choice of :math:`\mu<0` means that the utility :math:`u(.)` is always negative
(bounded by zero from above, i.e., for :math:`c\rightarrow \infty`), and the higher the argument :math:`U` in the exponential,
the lower the overall utility. A positive utility, :math:`U(.)`,  for :math:`l = P/F` implies that working reduces
the utility of consumption and that consumption and labor supply are complements.


Budget constraint
^^^^^^^^^^^^^^^^^

In a more involved case, the value function is maximised subject to the following budget constraint:

.. math::

	\begin{cases}
	a_{t+1} = (1+r)a_t + h_t w_t + m_t \tilde{h_t} \tilde{w_t} - T(l_t, X_t) - Q(t^k, h_t, \tilde{h_t}, m_t) - c_t,
	\\[4pt]
	a_{t+1} = \underline{a_s},
	\end{cases}

with initial and terminal conditions :math:`a_0 = 0` and :math:`a_{\tilde{t}+1} \geq 0`.

Notation is to be read as follows:

* :math:`r` - risk free interest rate
* :math:`(w, \tilde{w})` - hourly rates of wife and husband
* :math:`(h, \tilde{h})` - working hours of wife and husband
* :math:`\underline{a_s}` - borrowing limit, which is either zero, or equal to the amount of student loan borrowed (negative number)
* :math:`T` - tax and welfare transfer system, non-concave, non-smooth, and often discontinuous
* :math:`Q` - childcare costs

In the current simplified version of the model, the budget constrained is given by :math:`c_t =  h_t w_t + m_t \tilde{h_t} \tilde{w_t} - T(l_t, X_t) - Q(t^k, h_t, \tilde{h_t}, m_t)`.


Female wage equation
^^^^^^^^^^^^^^^^^^^^

The baseline specification of the female wage process is summarized in the following equations:

.. math::

	ln \hspace{2pt} w_t^m & = \gamma_{s,0}  + \gamma_{s,1} ln(e_t + 1) + \xi_t,\\
	ln \hspace{2pt} w_t & = ln \hspace{2pt} w_t^m - \xi_t,\\
	e_t & = e_{t-1}(1-\delta_s) + g_s(l_{t-1}),\\

where

* :math:`ln \hspace{2pt} w_t^m` - observed hourly wage rate
* :math:`\xi_t` - i.i.d. normal measurement error
* :math:`e_t` - experience measured in years
* :math:`\delta_s` - per period depreciation rate
* :math:`g_s` - per period rate of experience accumulation: :math:`g_s(F) = 1`


To be implemented
-----------------

The goal of this project is to develop a computational model similar to the one used in Blundell et. al. (2016).
Features of the model that are still missing in the current implementation include:

* budget constraint:

  * male wages
  * tax function which varies by year
  * childcare costs
  * savings
* female wage equation:

  * individual AR1 peoductivity process

* exogenous processes

  * male wage equation
  * probability of child arriving
  * probability of partner arriving
  * probability of partner leaving

Furthermore, we plan to include model features that go beyond the application in Blundell et. al. (2017):

* beliefs in the female wage equation
* labor market frictions

