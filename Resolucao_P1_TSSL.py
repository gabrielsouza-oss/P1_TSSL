# -*- coding: utf-8 -*-
# Primeiro configura printing com Latex
import sympy as sp
import numpy as np
import matplotlib.pyplot as mpl
from sympy.interactive import printing
from IPython.display import display, Math
from sympy import symbols
printing.init_printing(use_latex = True)

def round_expr(expr, num_digits):
  return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})

"""# **P1 - Transformadas**"""

mpl.rcParams["figure.dpi"] = 100
# Atualizar parâmetros matplotlib para Latex
mpl.rcParams.update({'font.size': 14, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

def resposta_estado_nulo(x_t,h_t):
  display(Math("\n h(t) ={}" + sp.latex(h_t)), Math("x(t) ={}" + sp.latex(x_t)))

  expr = sp.Integral(x_t.subs(t,j)*h_t.subs(t,t-j),(j,0,t))
  display(expr)

  result = expr.doit()
  result = round_expr(result,3)

  # Dando o print de expr em Latex 
  display(Math("\ny_h(t) ={}" + sp.latex(result) + " , t\geq0"))
  return result

"""**Passo 1 - Modelando o Sinal de entrada através da Série de Fourier Contínua no Tempo**

$$x(t) = 1, se -Ts<=x<=Ts$$
$$x(t) = 0, otherwise$$

"""

t , j , k , u, w = symbols('t j k u w') 
T0 = 4
Ts = 1
w0 = 2*sp.pi/T0
x = 1
auto = sp.exp(-sp.I*k*w0*t)

"""$$X_{k}=\frac{1}{T_{0}} \int_{T_{0}} x(t) e^{-j k \omega_{0} t} d t$$"""

Xk = (1/T0)*sp.Integral(x*auto,(t,-Ts,Ts))
Xk

Xks = sp.simplify(sp.combsimp(Xk.doit()))
Xks

# Série de Fourier Trigonométrica
x = sp.Function("x")
x = 0
#X0
x =  Xks.subs(k,0)
#Somatória
for i in range(1,100):
  x += ( 2*sp.re(Xks.subs(k,i))*sp.cos(k*w0*t) - 2*sp.im(Xks.subs(k,i))*sp.sin(k*w0*t)).subs(k,i)
x = round_expr(x,3)

# Série de Fourier Exponencial
#x = sp.Function("x")
#x = 0
#for i in range(-50,50):
  #x += Xks.subs(k,i)*auto.subs(k,i)
#x = round_expr(x,3)

lam_x = sp.lambdify(t, x, modules=['numpy'])
lam_x

x_vals = np.linspace(-T0-Ts-1, T0+Ts+1, 100)
y_vals = (abs(lam_x(x_vals)))
mpl.plot(x_vals, y_vals)
mpl.ylabel("v(t)")
mpl.xlabel("t")
mpl.title("v(t) x t")
mpl.savefig("v_100.jpg")
mpl.show()

#Espectro de Amplitude
x_vals = np.linspace(0, 20, 21)
y_vals = []

for i in x_vals:
  y_vals.append(abs(Xks.subs(k,i))) 

mpl.stem(x_vals, y_vals)
mpl.ylabel("|X(k)|")
mpl.xlabel("k")
mpl.title("Espectro de Amplitude sinal x(t)")
mpl.show()

#Espectro de Fase
x_vals = np.linspace(0, 20,21)
y_vals = []

for i in x_vals:
  y_vals.append( -sp.atan(( 2*sp.im( Xks.subs(k,i)) )/(2*sp.re(Xks.subs(k,i)))) )
  if y_vals[len(y_vals)-1] == sp.nan:
    y_vals[len(y_vals)-1] = -np.pi

mpl.stem(x_vals, y_vals,use_line_collection=True)
mpl.ylabel("θ(k)")
mpl.xlabel("k")
mpl.title("Espectro de Fase x(t)")
mpl.show()

"""**Passo 2 - Encontrando a reposta ao impulso unitário do sistema**

**Edo da Corrente do Sistema i(t)**

\begin{equation}
\begin{array}{r}
(D+1 / R C) I(t)= (1 / R) Dx(t) \\
\text { com } R=0.8 \Omega \text { e } C=0.1 F
\end{array}
\end{equation}
"""

#Definindo os parâmetros inciais do sistema 
R = 0.8
C = 0.1

y_n = sp.Function('y_n',real=True)
h_t = sp.Function('h_t',real=True)

#Definindo a equação para encontrarmos y_n(t)
eq = sp.Eq(sp.diff(y_n(t), t,1) + (1/(R*C))*y_n(t),0)
display(eq)

#Resolvendo a equação com as condições iniciais da função de DIRAC (Impulso Unitário)
res = sp.dsolve(eq,hint="best")
y_n = res
display(y_n)

#Substituindo as cond iniciais do sistema

const = sp.solve(sp.Eq(y_n.rhs.subs(t,0),1))[0]
y_n = y_n.rhs.subs("C1",const)

#Aplicar o operador P(D) [Termo que multiplica o termo x(t)] no resultado obtido em y_n
h_t =  (1/(R))*sp.diff(y_n,t)
h_t = round_expr(h_t,3)
display(h_t)

h_t = sp.simplify(sp.combsimp(h_t.doit())) + (1/(R))*sp.DiracDelta(t)
display(Math("h_t(t) ={}"+ sp.latex(h_t) + " , t\geq0"))

I_t = sp.Function("I_t")
I_t = resposta_estado_nulo(x,h_t)

corrente = sp.lambdify(t, I_t, modules = [{'Heaviside': lambda x: np.heaviside(x,1)}, 'numpy'] )
x_vals = np.linspace(0, 10, 1000)
y_vals = (corrente(x_vals))
mpl.plot(x_vals, y_vals)
mpl.ylabel("I(t)")
mpl.xlabel("t")
mpl.title("I(t) x t")
mpl.show()

"""**Reposta em Frequência I(t)**

\begin{equation}
H(j \omega)=\int_{-\infty}^{\infty} h(\tau) e^{-j \omega \tau} d \tau
\end{equation}
"""

I_t = 0 
X_k_til= 0
for i in range(1,100):
  I_t += (Xks.subs(k,i)*((sp.I*k*w0/R)/ (sp.I*k*w0+ (1/(R*C))) )*sp.exp(sp.I*k*w0*t)).subs(k,i)
  X_k_til += (Xks*((sp.I*k*w0/R)/ (sp.I*k*w0+ (1/(R*C))) ))
I_t = round_expr(I_t,3)

#Espectro de Amplitude
x_vals = np.linspace(0, 20, 21)
y_vals = []

for i in x_vals:
  y_vals.append(abs(X_k_til.subs(k,i))) 

mpl.stem(x_vals, y_vals)
mpl.ylabel("|X(k)|")
mpl.xlabel("k")
mpl.title("Espectro de Amplitude sinal I(t)")
mpl.show()

#Espectro de Fase
x_vals = np.linspace(0, 20,21)
y_vals = []

for i in x_vals:
  y_vals.append( -sp.atan( (2*sp.im( X_k_til.subs(k,i))) / (2*sp.re(X_k_til.subs(k,i))) ) )
  
mpl.stem(x_vals, y_vals,use_line_collection=True)
mpl.ylabel("θ(k)")
mpl.xlabel("k")
mpl.title("Espectro de Fase I(t)")
mpl.show()

"""**EDO da Tensão sobre o Capacitor $V_c(t)$**

\begin{equation}
\begin{array}{r}
(D+1 / R C) v_{c}(t)=(1 / R C) x(t) \\
\text { com } R=0.8 \Omega \text { e } C=0.1 F
\end{array}
\end{equation}
"""

y_n = sp.Function('y_n')
h_t = sp.Function('h_t')

#Definindo a equação para encontrarmos y_n(t)
eq = sp.Eq(sp.diff(y_n(t), t,1) + (1/(R*C))*y_n(t),0)
display(eq)
display(sp.latex(eq))

#Resolvendo a equação com as condições iniciais da função de DIRAC (Impulso Unitário)
res = sp.dsolve(eq,hint="best")
y_n = res
display(y_n)
display(sp.latex(y_n))

#Substituindo as cond iniciais do sistema

const = sp.solve(sp.Eq(y_n.rhs.subs(t,0),1))[0]
display(const)

y_n = y_n.rhs.subs("C1",const)

#Aplicar o operador P(D) [Termo que multiplica o termo x(t)] no resultado obtido em y_n
h_t =  (1/(R*C))*y_n
display(h_t)
display(sp.latex(h_t))
h_t = sp.simplify(sp.combsimp(h_t.doit())) + (1/(R*C))*sp.DiracDelta(t)
display(Math("h_t(t) ={}"+ sp.latex(h_t) + " , t\geq0"))
display(Math("h_t(t) ={}"+ sp.latex(h_t) + " , t\geq0").data)

V_c = sp.Function("V_c")
V_c = resposta_estado_nulo(x,h_t)

tensao = sp.lambdify(t, V_c, modules = [{'Heaviside': lambda x: np.heaviside(x,0)}, 'numpy'] )
x_vals = np.linspace(0, 10, 1000)
y_vals = (abs(tensao(x_vals)))
mpl.plot(x_vals, y_vals)
mpl.ylabel("$V_C(t)$")
mpl.title("$V_C(t)$ x t")
mpl.xlabel("t")
mpl.show()

x_vals = np.linspace(0, 10, 1000)
y_vals = (abs(tensao(x_vals)))
mpl.plot(x_vals, y_vals,label = "$V_C(t)$", color='blue')
y_vals = (abs(lam_x(x_vals)))
mpl.plot(x_vals, y_vals,label = "$V(t)$", color='red')
y_vals = (corrente(x_vals))
mpl.plot(x_vals, y_vals,label = "$I(t)$", color='green')
mpl.legend(loc ='lower right')
mpl.ylabel("$V_C(t)$")
mpl.title("Todos sinais x tempo")
mpl.xlabel("t")
mpl.savefig('comp.pdf', dpi=300) # para uso em Latex (Overleaf)
mpl.show()

"""# **Questão 4**

**Parte 4 - Sinal de entrada:
A = 2, T0 = 5, TH = 2 e
TL = 3.**
"""

t , j , k , u, w = symbols('t j k u w') 
T0 = 5
Ts = 2
w0 = 2*sp.pi/T0
x = 2
auto = sp.exp(-sp.I*k*w0*t)

Xk = (1/T0)*sp.Integral(x*auto,(t,0,Ts))
Xk

Xks = sp.simplify(sp.combsimp(Xk.doit()))
Xks

# Série de Fourier Trigonométrica
x = sp.Function("x")
x = 0
#X0
x =  Xks.subs(k,0)
#Somatória
for i in range(1,100):
  x += ( 2*sp.re(Xks.subs(k,i))*sp.cos(k*w0*t) - 2*sp.im(Xks.subs(k,i))*sp.sin(k*w0*t)).subs(k,i)
x = round_expr(x,3)

lam_x = sp.lambdify(t, x, modules=['numpy'])
lam_x

x_vals = np.linspace(-2, 3*T0-1, 1000)
y_vals = (abs(lam_x(x_vals)))
mpl.plot(x_vals, y_vals)
mpl.ylabel("V(t)")
mpl.xlabel("t")
mpl.title("V(t) x t")
mpl.show()

#Espectro de Amplitude
x_vals = np.linspace(0, 20, 21)
y_vals = []

for i in x_vals:
  y_vals.append(abs(sp.N(Xks.subs(k,i)))) 

mpl.stem(x_vals, y_vals)
mpl.ylabel("|X(k)|")
mpl.xlabel("k")
mpl.title("Espectro de Amplitude sinal V(t)")
mpl.show()

#Espectro de Fase
x_vals = np.linspace(0, 20,21)
y_vals = []

for i in x_vals:
  y_vals.append( sp.atan(( 2*sp.im( Xks.subs(k,i)) )/(2*sp.re(Xks.subs(k,i)))) )
  if y_vals[len(y_vals)-1] == sp.nan:
    y_vals[len(y_vals)-1] = 0

mpl.stem(x_vals, y_vals,use_line_collection=True)
mpl.ylabel("θ(k)")
mpl.xlabel("k")
mpl.title("Espectro de Fase V(t)")
mpl.show()

"""**Edo da Corrente do Sistema i(t)**

\begin{equation}
\begin{array}{r}
(D+1 / R C) I(t)= (1 / R) Dx(t) \\
\text { com } R=0.8 \Omega \text { e } C=0.1 F
\end{array}
\end{equation}
"""

#Definindo os parâmetros inciais do sistema 
R = 0.8
C = 0.1

y_n = sp.Function('y_n',real=True)
h_t = sp.Function('h_t',real=True)

#Definindo a equação para encontrarmos y_n(t)
eq = sp.Eq(sp.diff(y_n(t), t,1) + (1/(R*C))*y_n(t),0)
display(eq)

#Resolvendo a equação com as condições iniciais da função de DIRAC (Impulso Unitário)
res = sp.dsolve(eq,hint="best")
y_n = res
display(y_n)

#Substituindo as cond iniciais do sistema

const = sp.solve(sp.Eq(y_n.rhs.subs(t,0),1))[0]
y_n = y_n.rhs.subs("C1",const)

#Aplicar o operador P(D) [Termo que multiplica o termo x(t)] no resultado obtido em y_n
h_t =  (1/(R))*sp.diff(y_n,t)
h_t = round_expr(h_t,3)
display(h_t)
h_t = sp.simplify(sp.combsimp(h_t.doit())) + (1/(R))*sp.DiracDelta(t)
display(Math("h_t(t) ={}"+ sp.latex(h_t) + " , t\geq0"))

I_t = sp.Function("I_t")
I_t = resposta_estado_nulo(x,h_t)

corrente = sp.lambdify(t, I_t, modules = [{'Heaviside': lambda x: np.heaviside(x,1)}, 'numpy'] )
x_vals = np.linspace(0, 10, 1000)
y_vals = (corrente(x_vals))
mpl.plot(x_vals, y_vals)
mpl.ylabel("I(t)")
mpl.xlabel("t")
mpl.title("I(t) x t")
mpl.show()

"""**Reposta em Frequência I(t)**

\begin{equation}
H(j \omega)=\int_{-\infty}^{\infty} h(\tau) e^{-j \omega \tau} d \tau
\end{equation}
"""

I_t = 0 
X_k_til= 0
for i in range(1,100):
  I_t += (Xks.subs(k,i)*((sp.I*k*w0/R)/ (sp.I*k*w0+ (1/(R*C))) )*sp.exp(sp.I*k*w0*t)).subs(k,i)
  X_k_til += (Xks*((sp.I*k*w0/R)/ (sp.I*k*w0+ (1/(R*C))) ))
I_t = round_expr(I_t,3)

#Espectro de Amplitude
x_vals = np.linspace(0, 20, 21)
y_vals = []

for i in x_vals:
  y_vals.append(abs(sp.N(X_k_til.subs(k,i)))) 

mpl.stem(x_vals, y_vals)
mpl.ylabel("|X(k)|")
mpl.xlabel("k")
mpl.title("Espectro de Amplitude sinal I(t)")
mpl.show()

#Espectro de Fase
x_vals = np.linspace(0, 20,21)
y_vals = []

for i in x_vals:
  y_vals.append( -sp.atan( (2*sp.im( X_k_til.subs(k,i))) / (2*sp.re(X_k_til.subs(k,i))) ) )
  if y_vals[len(y_vals)-1] == sp.nan:
    y_vals[len(y_vals)-1] = 0

mpl.stem(x_vals, y_vals,use_line_collection=True)
mpl.ylabel("θ(k)")
mpl.xlabel("k")
mpl.title("Espectro de Fase I(t)")
mpl.show()

"""**EDO da Tensão sobre o Capacitor $V_c(t)$**

\begin{equation}
\begin{array}{r}
(D+1 / R C) v_{c}(t)=(1 / R C) x(t) \\
\text { com } R=0.8 \Omega \text { e } C=0.1 F
\end{array}
\end{equation}
"""

y_n = sp.Function('y_n')
h_t = sp.Function('h_t')

#Definindo a equação para encontrarmos y_n(t)
eq = sp.Eq(sp.diff(y_n(t), t,1) + (1/(R*C))*y_n(t),0)
display(eq)

#Resolvendo a equação com as condições iniciais da função de DIRAC (Impulso Unitário)
res = sp.dsolve(eq,hint="best")
y_n = res
display(y_n)

#Substituindo as cond iniciais do sistema

const = sp.solve(sp.Eq(y_n.rhs.subs(t,0),1))[0]
display(const)

y_n = y_n.rhs.subs("C1",const)

#Aplicar o operador P(D) [Termo que multiplica o termo x(t)] no resultado obtido em y_n
h_t =  (1/(R*C))*y_n
display(h_t)
h_t = sp.simplify(sp.combsimp(h_t.doit())) + (1/(R*C))*sp.DiracDelta(t)
display(Math("h_t(t) ={}"+ sp.latex(h_t) + " , t\geq0"))

V_c = sp.Function("V_c")
V_c = resposta_estado_nulo(x,h_t)

tensao = sp.lambdify(t, V_c, modules = [{'Heaviside': lambda x: np.heaviside(x,0)}, 'numpy'] )
x_vals = np.linspace(0, 10, 1000)
y_vals = (abs(tensao(x_vals)))
mpl.plot(x_vals, y_vals)
mpl.ylabel("$V_C(t)$")
mpl.title("$V_C(t)$ x t")
mpl.xlabel("t")
mpl.show()

x_vals = np.linspace(0, 10, 1000)
y_vals = (abs(tensao(x_vals)))
mpl.plot(x_vals, y_vals,label = "$V_C(t)_n$", color='blue')
y_vals = (abs(lam_x(x_vals)))
mpl.plot(x_vals, y_vals,label = "$V(t)_n$", color='red')
y_vals = (corrente(x_vals))
mpl.plot(x_vals, y_vals,label = "$I(t)_N$", color='green')
mpl.legend(loc ='lower right')
mpl.title("Todos sinais x tempo")
mpl.xlabel("t")
mpl.savefig('comp_2.pdf', dpi=300) # para uso em Latex (Overleaf)
mpl.show()