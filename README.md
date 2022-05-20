# SimpleMemory
A very simple model of human memory retention

$$
E = m c^2
$$
A simple model of memory 
The purpose of this write-up is to design an algorithm that optimizes the learning of a collection of items. Specifically, to maximize recall with minimal effort while assuming an opportunistic study schedule.
Notation: 
	i indexes the collection of items that need to be learned
	j indexes successive tests for the same item
	p is the probability that an item will be remembered
	t is the time passed since (re-)learning an item
	r is the pass/fail result of a test
	γ is the rate of forgetting
	δ is the rate of learning

PART 1
We assume that an item that has been learned will be forgotten over time and that ‘forgetting’ is a random process with constant rate.  The probability p that an item is remembered at time t can then be modelled as:
▭( p(t)=e^(-γt)  )
The parameter γ is on the order of 0.1 when t is measured in days , but γ depends on the complexity of the item and on the aptitude of the student, which are both unknown.  Note that when γ=0 there is no ‘forgetting’.
Memory loss can be counter-acted by repeatedly reviewing  previously learned items.  With this approach, when an item is successfully remembered, its memory decay rate is assumed to be reduced according the model function:
γ_(j+1)=γ_j*e^δ(p_j-1) 
The parameter δ is on the order of 2.0 but, just like γ, depends on the complexity of the item and on the aptitude of the student. Note that when δ=0 there is no ‘learning’. The formula implies that delayed testing will increase long-term retention.
When a student fails to remember an item, we still consider this a leaning moment and we assume the item is re-learned  at decay rate γ_j.  
In summary, testing an item changes its memory decay rate as follows:
▭(〖 γ〗_(j+1)={█(〖 γ〗_j*e^δ(p_j-1) ,&r_j=1@γ_j,&r_j=0 )┤ )

What happens to memory when an item is tested?
The probability to successfully remember an item is given by p(t).  We can define memory M as the ability to remember an item in the future, so that M is the time integral of p(t):
M(t)=∫_t^∞▒〖p(t)dt=∫_t^∞▒〖e^(-γt) dt=[-1⁄γ*e^(-γt) ]_t^∞ 〗=(p(t))⁄γ〗
Right after learning an item, its memory M has the maximum value of:
M(0)=(p(0))⁄γ=1⁄γ
For newly learned items the increase in memory is:
▭( ∆M_0=1⁄(γ_0  ))
If an item is tested and remembered, its memory decay rate will be reduced. The net memory change is the difference between the memory associated with the new, slower decay rate and the remaining memory at the previous, faster decay rate:
〖∆M〗_([r=1] )=M_(j+1) (0)-M_j (t)
=1⁄γ_(j+1) -p_j⁄γ_j 
Similarly, the memory change for an unsuccessful recall is:
〖∆M〗_[r=0] =M_j (0)-M_j (t)
=1⁄γ_j -p_j⁄γ_j 
For each test, the likely net memory increase can be calculated as the probability-weighted sum of ∆M for the two possible outcomes:
∆M=〖p_([r=1])*∆M〗_([r=1])+〖p_([r=0])*∆M〗_([r=0])
=p_j*(1⁄γ_(j+1) -p_j⁄γ_j )+(1-p_j )*(1⁄γ_j -p_j⁄γ_j )
=p_j⁄γ_(j+1) -(p_j p_j)⁄γ_j +1⁄γ_j -p_j⁄γ_j -p_j⁄γ_j +(p_j p_j)⁄γ_j 
Giving:
▭( ∆M=p_j⁄γ_(j+1) +(1-2 (p_j))⁄γ_j   )
The learning algorithm is to repeatedly test the items with the highest ∆M unless learning a new item adds more memory.
To illustrate let’s pick j=0, δ=2 and γ_0=0.2.  Learning a new item then becomes worthwhile when
∆M_0>∆M 
1⁄γ_0 >p⁄γ_1 +(1-2 (p))⁄γ_0 
5>5pe^(-2(p-1) )+5-10p
p> ≈0.65
PART 2
How to determine the actual values of γ and δ ?
We have reasonable initial estimates of γ and δ from the literature, but the algorithm would work best with values that are specific for the individual student. When enough items have been learned and tested it may be possible to calculate a personal value for γ from the test results.  And when enough tests have been repeated it may be possible to also derive a personal value for δ. Let’s first look at γ in a simplified case.
We can estimate a model parameter from observed memory tests using regression analysis. The goal is to maximize the likelihood that the model predicts the observed outcomes.  
The memory test is a binary (pass/fail) trial, implemented through e.g. a multiple-choice test with outcome r=1 when the test was a pass or outcome r=0 when the test was a fail.  The probability that an item will be remembered is:
p_([r=1])=e^(-γt)
and the probability that it will not be remembered is therefore:
p_([r=0])=1-e^(-γt)
To calculate the value of γ that best fits the results we look at a set of tests that represent a first review of already learned items.  We can represent this as:
(r_0,t_0 ),(r_1,t_1 ),⋯,(r_n,t_n )
We assume that all items are similar and that the tests are independent, so that γ is a valid model parameter for the entire set of tests. The likelihood for the test results to be what they are, is then given by the product of the individual item probabilities p_i:
L= ∏_(i=1)^n▒p_i =∏_(r_i=1)▒e^(-γt_i )  ∏_(r_i=0)▒(1-e^(-γt_i ) ) 
The log-likelihood is:
LL=∑_(r_i=1)▒ln⁡(e^(-〖γt〗_i ) ) +∑_(r_i=0)▒ln⁡(1-e^(-〖γt〗_i ) ) 
=∑_(r_i=1)▒〖-γt_i 〗+∑_(r_i=0)▒ln⁡(1-e^(-〖γt〗_i ) ) 
The value of γ that maximizes LL (and therefore maximizes L), can be found using the first and second order partial derivatives:
∂LL/∂γ=∑_(〖 r〗_i=1)▒〖-t〗_i +∑_(r_i=0)▒〖t_i  e^(-〖γt〗_i )/(1-e^(-〖γt〗_i ) )〗
(∂^2 LL)/〖∂γ〗^2 =∑_(r_i=0)▒(-〖t_i〗^2 e^(-γt_i ))/(1-e^(-γt_i ) )^2 
Because the second derivative of LL is always negative, LL is concave and its maximum can be found where the first derivative is zero, using e.g. Ridder’s method . 
We cannot determine δ by only looking at first reviews, so we must now take the full repetition history into account.
How to determine the actual value of δ ?
The probability that an item is remembered not only depends on the time since the last test but on its complete test history. Clearly p_(j+1) is a function of p_j which is a function of p_(j-1) and so on:
p_(j+1)=e^(-γ_(j+1) t)=e^(-γ_j*t*e^δ(p_j-1)  )
We can find the values of γ and δ by maximizing the likelihood that the full test history is what it is. That full history can be represented by an irregular matrix of elements (r_ij,t_ij )  where i denotes the items and j indexes the successive tests for that item.  The likelihood that the whole test series is what it is, then becomes (with limit m varying for every value of i):
L= ∏_(i=1)^n▒∏_(j=1)^(m_i)▒p_ij =∏_(r_ij=1)▒e^(-γ_ij t_ij )  ∏_(r_ij=0)▒(1-e^(-γ_ij t_ij ) ) 
The log-likelihood is:
LL=∑_(r_ij=1)▒〖-〖γ_ij t〗_ij 〗+∑_(r_ij=0)▒ln⁡(1-e^(-〖γ_ij t〗_ij ) ) 
The partial derivative of LL to γ_i0 is:
∂LL/(∂γ_i0 )=∑_(〖 r〗_ij=1)▒〖-t_ij  (∂γ_ij)/(∂γ_i0 )〗+∑_(r_ij=0)▒〖t_ij  e^(-〖γ_ij t〗_ij )/(1-e^(-〖γ_ij t〗_ij ) )  (∂γ_ij)/(∂γ_i0 )〗
We again assume that all items in a collection are similar, so that γ_i0 has the same value for every i.  We can then use γ_0 as the model parameter for the entire collection of items and for the whole set of tests. 
Dropping the i index, we can calculate (∂γ_ij)/(∂γ_i0 ) for each item using the chain rule:
(∂γ_j)/(∂γ_0 )=(∂γ_j)/(∂γ_(j-1) )*(∂γ_(j-1))/(∂γ_(j-2) )*…*(∂γ_1)/(∂γ_0 )=∏_(z=0)^(j-1)▒(∂γ_(z+1))/(∂γ_z )
And calculate each term as:
(∂γ_(z+1))/(∂γ_z )=∂(γ_z*e^δ(p_z-1)  )/(∂γ_z )
=e^δ(p_z-1) +γ_z*(∂e^δ(p_z-1) )/(∂γ_z )
=e^δ(p_z-1) +γ_z*e^δ(p_z-1) *∂(δ(p_z-1))/(∂γ_z )
=e^δ(p_z-1) +γ_z*e^δ(p_z-1) *δ*(∂p_z)/(∂γ_z )
=e^δ(p_z-1) +γ_z*e^δ(p_z-1) *δ*∂(e^(-〖γ_z t〗_z ) )/(∂γ_z )
=e^δ(p_z-1) -γ_z 〖*e〗^δ(p_z-1) *δ*t_z*p_z
=〖(1-〖δ*γ〗_z*t_z*p_z )*e〗^δ(p_z-1) 
≡A_z
Introducing a shorthand notation:
G_j≡(∂γ_j)/(∂γ_0 )=∏_(z=0)^(j-1)▒(∂γ_(z+1))/(∂γ_z )=∏_(z=0)^(j-1)▒A_z 
Results in:
▭( ∂LL/(∂γ_0 )=∑_(〖 r〗_ij=1)▒〖-t_ij G_ij 〗+∑_(r_ij=0)▒〖t_ij  e^(-〖γ_ij t〗_ij )/(1-e^(-〖γ_ij t〗_ij ) ) G_(ij ) 〗)

The partial derivative of LL to δ is:
∂LL/∂δ=∑_(〖 r〗_ij=1)▒〖-t_ij  (∂γ_ij)/∂δ〗+∑_(r_ij=0)▒〖t_ij  e^(-〖γ_ij t〗_ij )/(1-e^(-〖γ_ij t〗_ij ) )  (∂γ_ij)/∂δ〗
We can calculate (∂γ_j)/∂δ recursively starting with (∂γ_0)/∂δ=0 and:
(∂γ_(j+1))/∂δ=∂(γ_j*e^δ(p_j-1)  )/∂δ
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*∂(e^δ(p_j-1)  )/∂δ
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*e^δ(p_j-1) *∂δ(p_j-1)/∂δ
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*e^δ(p_j-1) *((p_j-1)+δ ∂(p_j-1)/∂δ)
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*e^δ(p_j-1) *(p_j-1+δ (∂p_j)/∂δ)
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*e^δ(p_j-1) *(p_j-1+δ (∂e^(-〖γ_j t〗_j ))/∂δ)
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*e^δ(p_j-1) *(p_j-1-t_j δe^(-〖γ_j t〗_j )  (∂γ_j)/∂δ)
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*(p_j-1)*e^δ(p_j-1) -t_j δe^(-〖γ_j t〗_j ) γ_j*e^δ(p_j-1)   (∂γ_j)/∂δ
=e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j*(p_j-1)*e^δ(p_j-1) -t_j δγ_j p_j*e^δ(p_j-1)   (∂γ_j)/∂δ
=(1-δγ_j t_j p_j ) e^δ(p_j-1) *(∂γ_j)/∂δ+γ_j (p_j-1) e^δ(p_j-1) 
≡A_j  (∂γ_j)/∂δ+B_j
Introducing some more shorthand:
D_j≡(∂γ_j)/∂δ={█(0,&j=0@A_(j-1)  (∂γ_(j-1))/∂δ+B_(j-1),&j>0 )┤
Gives the result:
▭( ∂LL/∂δ=∑_(〖 r〗_ij=1)▒〖-t_ij D_ij 〗+∑_(r_ij=0)▒〖t_ij  e^(-〖γ_ij t〗_ij )/(1-e^(-〖γ_ij t〗_ij ) ) D_ij  〗)
A simulation of 100 items tested 100 times with γ=0.2 and δ=2.0 shows the following parameter sensitivity:
 
PART 3
Fletcher
 
Ford
 
Shanno
 NumericalRecipes 
 
AG
BFGS
 
DFP
 

PART 4
In part 1 we derived a formula to select the most valuable question. In part 2 we found a way to estimate the memory decay and learning rate parameters from a set of actual test results. Let's now focus our attention to building efficient tests.
How to generate multiple-choice tests?
Designing multiple choice questions is conceptually simple: given one right answer, pick a few alternatives as distractors. The job of the distractors is to force an explicit evaluation of all the alternatives, so they must be plausible.  Plausible distractors make the tests hard and the effort to find the right answer will increase learning.
So, what makes a distractor 'plausible'?  In the beginning all items are equal and there is no algorithmic way to determine the plausibility of potential distractors. The distractors will have to be chosen randomly or heuristically. 
As more tests are taken, we may start to see patterns in the results and detect that items are correlated.  The answer to question q1 should be answer a1, but sometimes a2 is selected instead.  This pattern becomes stronger when a1 is (wrongly) chosen for q2.  Answers a1 and a2 are confused for each other. 
These wrong answers are obvious candidates for distraction, so a2 is a good distractor for q1 and vice versa.  The obvious algorithm is to track all of the wrong answers for each question and then select the most frequently occurring wrong answers as distractors for that question. This a naïve approach because it could lead to sticky distractors.  As the frequency of mistakes goes down over time and old mistakes remain in the list, the distractors will settle.
Fortunately, even mistakes are forgotten.  Let's assume that the memory of a mistake is decaying just as fast as the memory of a newly learned item, and a repeated mistake changes the memory decay rate as explained earlier in part 1:

▭( p(t)=e^(-γt)  )
▭(〖 γ〗_(j+1)=〖 γ〗_j*e^δ(p_j-1)  )

Over time we can say that items that have been learned a long time ago will have been forgotten and that these are more plausible distractors than more recently learned items.  From part 1 we can deduce that plausibility is inversely proportional to p.

Gradually phase in more difficulty. Can't start with most difficult distractors.  Start with unlikely distractors.  When is learning enough to become more tough ?
Random (before 'learned), heuristics (before mistakes)
How do the heuristics nudge mistakes ?  Select a-priory distractors and bring them in at a moment that leaning is not yet established. Heuristics die off as they are answered correctly.
Balance sense-of-progress against sense-of-accomplishment
Individual/relative item difficulty
Average person versus individual results
