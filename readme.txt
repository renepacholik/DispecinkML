FORMÁT VSTUPNÍHO SOUBORU:
	header: "Urgentnost"\t"OrigTest"
	data:   int úroveňUrgentnosti\t"textZprávy"

příklad:
Urgentnost	OrigText
2	Dobrý den,na zítra tj. 23.7.2020 jsem měla domluvenou dálkovou správu v čase 9,00 - 10,00 hod. p. Vorlíček. Prosím o zrušení. Děkuji
2	Prosím přenesení vyfiltrovaných dětí z jedné školky (programu) do druhé školky. děkujiBartůňková
1	Dobrý den, potřebuju zrušit měsíční uzávěrku, nejde mi provést inkaso, protože u jednoho strávníka nebyl variabilní symbol. Děkuji Jansová

POJMY:

	Trainer - Je výsledkem součtu algoritmu a úkolu. Algoritmy můžeme používat pro několik účelů, a to binární klasifikaci, vícetřídní klasifikaci a regresi. Rozdíl je v tom, jak výstup algoritmu interpretujeme, aby odpovídal úkolu. Pro každou kombinaci algoritmu/úkolu poskytuje ML.NET komponent, který vykoná trénovací algoritmus jež provede interpretaci. Tyto komponenty se nazývají trenéři. 
	Model - je výsledný algoritmus, který určuje, do jaké třídy vstupní hodnota patří. Je to výsledkem práce trenéra s daty.
	Micro accuracy - Zahrne příspěvky všech tříd pro výpočet průměrné metriky. Je to zlomek správně předpovězených případů. Čím více se hodnota blíží k číslu 1,00, tím přesnější je výsledek. Ve vícetřídní klasifikaci je micro-accuracy lepší než macro-accurcy, když předpokládáte, že bude existovat nerovnost mezi počtem dat pro jednotlivé třídy. 
	Macro accuracy - Vypočítá přesnost všech tříd a průměrem těchto přesností je právě macro-accuracy. Čím blíže je hodnota položena k číslu 1,00 tím lépe. Macro-accuracy počítá metriku nezávisle na každé třídě, tím pádem zachází se všemi třídami stejně. 
	Log loss - Log-loss se zvyšuje s tím, jak se předpokládaná pravděpodobnost odchyluje od skutečného označení. Čím více se přibližuje hodnotě 0,00 tím lepší. Perfektní model by měl log-loss 0.00, cílem je tedy tuto hodnotu co nejvíce přiblížit minimální hodnotě, která je pro nás nejlepší. 
	Log loss reduction - Vypočítává míru toho, jak moc se model zlepšil oproti modelu, který dává náhodné předpovědi. Hodnota 1.00 zde označuje maximální míru zlepšení, tudíž se jedná o hodnotu pro nás nejlepší.
	 
	Confusion table - Matice záměn, ve sloupcích je skutečná hodnota a v řádcích předpověď. 
	Precision - Poměr správných pozitivních předpovědí ve vztahu k celku pozitivních předpovědí. 
	Recall - Poměr správných pozitivních předpovědí ve vztahu k celkovým pozitivním příkladům.

         	  ||========================
	PREDICTED ||     0 |     1 |     2 | Recall
	TRUTH     ||========================
      	  	0 || 1,382 |    17 |    15 | 0.9774
     	        1 ||    32 |   591 |    11 | 0.9322
        	2 ||    28 |     8 |   614 | 0.9446
          	  ||========================
	Precision ||0.9584 |0.9594 |0.9594 |

	Zde můžeme vidět, že příklady označené urgentností 0 byly správně určené jako '0' 1382x, špatně určené jako '1' 32x a jako '2' 28x.

	Více informací: https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/metrics