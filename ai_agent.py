import logging
import traceback
import os
from typing import Tuple, Optional
from agents import Agent, Runner
from models import Quiz
from utils import save_text_to_file

class QuizGenerator:
    """Class for generating quizzes using AI agents"""
    
    def __init__(self, model: str, summary_dir: str):
        """Initialize the quiz generator
        
        Args:
            model (str): The OpenAI model to use
            summary_dir (str): Directory to save summaries
        """
        self.model = model
        self.summary_dir = summary_dir
        os.makedirs(self.summary_dir, exist_ok=True)
    
    async def create_quiz_from_text(self, text: str, filename: str, language: str, num_questions: int = 10) -> Tuple[Optional[Quiz], Optional[str]]:
        """Process a single text document through the agent pipeline
        
        Args:
            text (str): The text to process
            filename (str): The name of the file to process
            language (str): The language to generate the quiz in
            num_questions (int): Number of questions to generate (default: 10)

        Returns:
            Tuple[Optional[Quiz], Optional[str]]: A tuple containing the quiz and the filename
        """
        try:
            # remove .pdf extension from filename
            base_filename = filename.replace('.pdf', '')
            
            # processing with summarizer agent
            summarizer = Agent(
                name="text summarizer",
                instructions=f"""
                sei un esperto nella creazione di riassunti dettagliati di testi medici sulle malattie.
                crea un riassunto completo che catturi tutte le informazioni importanti sulla/e malattia/e descritta/e nel testo.
                concentrati sull'estrazione di informazioni mediche chiave come:
                - nomi e classificazioni delle malattie
                - sintomi e manifestazioni cliniche
                - cause e fattori di rischio
                - metodi diagnostici
                - approcci terapeutici e gestione
                - strategie di prevenzione
                - epidemiologia e statistiche
                
                tutti i riassunti devono essere in {language}.
                mantieni l'accuratezza medica rendendo il contenuto più conciso.
                organizza le informazioni in modo strutturato che evidenzi gli aspetti più importanti della/e malattia/e.
                """,
                model=self.model
            )
            summary_result = await Runner.run(summarizer, text)
            
            # save summary
            summary_path = os.path.join(self.summary_dir, f"{base_filename}_summary.txt")
            save_text_to_file(summary_result.final_output, summary_path)
            
            # quiz generation
            quiz_generator = Agent(
                name="quiz generator",
                instructions=f"""
                sei un esperto nella creazione di quiz educativi specificamente sulle malattie.
                crea esattamente {num_questions} domande a scelta multipla basate sul testo fornito.
                
                IMPORTANTE: Concentrati SOLO sulla malattia descritta nel testo. Non andare fuori tema.
                Queste domande saranno utilizzate per un serious game, quindi l'accuratezza e la pertinenza sono fondamentali.
                
                per ogni domanda:
                1. identifica un aspetto specifico della malattia menzionata nel testo (sintomi, cause, trattamenti, ecc.)
                2. crea una domanda chiara e medicalmente accurata su quell'aspetto
                3. fornisci esattamente 4 risposte con questi punteggi:
                   - una risposta corretta (5 punti)
                   - una risposta quasi corretta (2 punti)
                   - una risposta sbagliata (0 punti)
                   - una risposta molto sbagliata (-2 punti)
                
                assegna questi valori in base all'accuratezza medica e al testo fornito.
                tutte le domande e le risposte devono essere in {language}.
                non fare domande sciocche o includere informazioni non correlate alla malattia nel testo.
                le domande NON devono menzionare il testo di riferimento.
                assicurati che ogni domanda abbia esattamente 4 risposte.
                """,
                output_type=Quiz,
                model=self.model
            )
            quiz_result = await Runner.run(quiz_generator, summary_result.final_output)
            
            return quiz_result.final_output_as(Quiz), base_filename  
            
        except Exception as e:
            logging.error(f"error processing {filename}: {str(e)}")
            logging.error(traceback.format_exc())
            return None, None