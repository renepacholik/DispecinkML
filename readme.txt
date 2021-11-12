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

OVLÁDÁNÍ:
	
	Config - errorLogPath a logPath by měli vést pouze ke složce kde se soubory budou ukládat. Jinak program uloží logy do složky Logs v aktuálním adresáři.
		 predictionPathType by měl obsahovat pouze ".txt" nebo ".json" podle toho jaký formát chcete u výsledného souboru. Pokud bude obsahovat jinou hodnotu nebo bude prázdný, výsledek se nevypíše. Pokud dáte do cesty uložení předpovědi soubor s koncovkou, program tuto koncovku smaže a nahradí jí koncovkou z configu.

		Pro přeskočení některých parametrů nahraďte hodnotu písmenem 'd' pro výchozí hodnotu. Toto má být použito, pokud se potřebujete dostat k argumentu, kterému předchází argument, jež nechcete změnit.
		př.: UrgentnostML p "Potřebuji akutní pomoc" d C:\Users\Dash\Desktop\test\UrgentnostML.zip

	Trénovaní: UrgentnostML t cestaKeZdrojiDat početIterací čísloTrenéra cestaKUloženíModeluANázev (Zde je povinný argumnet cestaKeZdrojiDat)

		   	cestaKeZdrojiDat - Cesta kde je uložený soubor s daty na trénování ve správném formátu viz. FORMÁT VSTUPNÍHO SOUBORU
			početIterací - Celé kladné číslo představující počet iterací, kterých má ternér udělat (Více iterací = delší doba trénování, ale přesnější model)
			čísloTrenéra - Číselné označení jména trenéra pro použítí k vytvoření modelu.(viz tabulka níže) Jako nejlepšího trenéra jsem vybral číslo 1 LightGbm.

				'1' - LightGbm
                   		'2' - AveragedPerceptron
                   		'3' - SdcaMaximumEntropy
                   		'4' - SymbolicSgdLogisticRegression
                  		'5' - LinearSvm
                  		'6' - SgdCalibrated
                  		'7' - SgdNonCalibrated

			cestaKUloženíModeluANázev - Cesta kam se uloží model po dokončení tréninku. V cestě musí být i název souboru s koncovkou .zip. Pokud cesta bude prázdná nebo neplatná program zvolí cestu sám a to do akutálního adresáře. Pokud nebude zadán název souboru končící .zip, bude nastaven výchozí název UrgentnostML.zip.

	      př.: UrgentnostML t C:\Users\Dash\Desktop\.NET_ML\ML_Dispecink\ML_Dispecink\UrgentnostML\Input\extracted_csv 120 2 C:\Users\Dash\Desktop\test\UrgentnostML.zip

	Předpovídání: UrgentnostML p textZprávy cestaKUloženíANázev cestaKModelu (Zde je povinný argument textzprávy)

			textZprávy - Zprává u které chcete zjistit urgentnost. Použijte formát: "text"
			cestaKUloženíANázev - Cesta kam se uloží vyhodnocení. V cestě musí být i název souboru s koncovkou .txt.okud cesta bude prázdná nebo neplatná program zvolí cestu sám a to do akutálního adresáře.. Pokud nebude zadán název souboru končící .txt, bude nastaven výchozí název out.txt. 
			cestaKModelu - Cesta kde je uložen model v .zip, který chcete použít pro vyhodnocení.

				
	      př.: UrgentnostML p "Potřebuji akutní pomoc" C:\Users\Dash\Desktop\test\out.txt C:\Users\Dash\Desktop\test\UrgentnostML.zip

   