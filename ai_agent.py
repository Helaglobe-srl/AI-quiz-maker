import logging
import traceback
import os
from typing import Tuple, Optional, List

from agents import Agent, Runner
from models import Quiz, Question
from utils import save_text_to_file

# estimate character limit for token limits:
# OpenAI's gpt-4.1 TPM is 30,000. A chunk of 15,000 chars is a good threshold.
MAX_CHUNK_SIZE = 15000

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

    def _split_text_into_chunks(self, text: str, chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Splits text into chunks of a specified size."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        logging.info(f"Split text into {len(chunks)} chunks.")
        return chunks

    async def _summarize_chunk(self, chunk: str, chunk_num: int, total_chunks: int, language: str, base_summarizer_instructions: str) -> str:
        """Summarizes a single chunk of text."""
        logging.info(f"Summarizing chunk {chunk_num + 1}/{total_chunks}...")
        contextual_prompt_addition = ""
        if total_chunks > 1:
            contextual_prompt_addition = f"\n\nNota: Questo è il frammento {chunk_num + 1} di {total_chunks} di un documento più grande. Concentrati sul riassumere questo specifico frammento nel contesto del suo potenziale collegamento con altri frammenti."
        
        current_instructions = base_summarizer_instructions + contextual_prompt_addition
        
        chunk_summarizer_agent = Agent(
            name=f"text_summarizer_chunk_{chunk_num + 1}",
            instructions=current_instructions,
            model=self.model
        )

        try:
            summary_result = await Runner.run(chunk_summarizer_agent, chunk)
            logging.info(f"Successfully summarized chunk {chunk_num + 1}/{total_chunks}.")
            return summary_result.final_output
        except Exception as e:
            logging.error(f"Error summarizing chunk {chunk_num + 1}/{total_chunks}: {str(e)}")
            logging.error(traceback.format_exc())
            return "" # empty string on error

    async def create_quiz_from_text(self, text: str, filename: str, language: str, num_questions_total: int = 10) -> Tuple[Optional[Quiz], Optional[str]]:
        """Process a single text document, generate questions per chunk, and aggregate them.
        
        Args:
            text (str): The text to process
            filename (str): The name of the file to process
            language (str): The language to generate the quiz in
            num_questions_total (int): Total number of questions desired for the quiz (default: 10)

        Returns:
            Tuple[Optional[Quiz], Optional[str]]: A tuple containing the final quiz and the base filename
        """
        try:
            base_filename = filename.replace('.pdf', '').replace('.txt', '')
            
            text_chunks = self._split_text_into_chunks(text)
            num_chunks = len(text_chunks)

            if num_chunks == 0:
                logging.warning(f"No text chunks to process for {filename}.")
                return None, base_filename

            all_questions: List[Question] = []
            aggregated_chunk_summaries_for_saving = []

            base_summarizer_instructions = f"""
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
            """

            for i, chunk in enumerate(text_chunks):
                logging.info(f"Processing chunk {i + 1}/{num_chunks} for {filename}...")
                # 1. summarize chunk
                chunk_summary = await self._summarize_chunk(chunk, i, num_chunks, language, base_summarizer_instructions)
                
                if not chunk_summary:
                    logging.warning(f"Chunk {i + 1}/{num_chunks} for {filename} could not be summarized. Skipping quiz generation for this chunk.")
                    continue

                aggregated_chunk_summaries_for_saving.append(chunk_summary)

                # determine questions for this chunk if total not yet met
                remaining_questions_needed = num_questions_total - len(all_questions)
                if remaining_questions_needed <= 0 and num_questions_total > 0:
                    logging.info(f"Target of {num_questions_total} questions reached. Stopping further question generation.")
                    break 
                
                if num_questions_total == 0:
                    logging.info("num_questions_total is 0, no questions will be generated.")
                    break

                remaining_chunks_to_process = num_chunks - i
                questions_to_attempt_for_this_chunk = max(1, (remaining_questions_needed + remaining_chunks_to_process - 1) // remaining_chunks_to_process) if remaining_chunks_to_process > 0 else 0
                
                if questions_to_attempt_for_this_chunk <= 0:
                    continue

                logging.info(f"Attempting to generate {questions_to_attempt_for_this_chunk} questions for chunk {i + 1}/{num_chunks} of {filename}.")

                # 2. generate quiz questions from this chunk's summary
                quiz_generator_agent_for_chunk = Agent(
                    name=f"quiz_generator_chunk_{i+1}",
                    instructions=f"""
                    Sei un esperto nella creazione di quiz educativi su malattie, destinati a persone che vivono con quella condizione.
                    Il testo fornito è un riassunto del frammento {i + 1} di {num_chunks} di un documento più grande.
                    Crea esattamente {questions_to_attempt_for_this_chunk} domande a scelta multipla basate ESCLUSIVAMENTE sul testo di QUESTO RIASSUNTO DEL FRAMMENTO.

                    IMPORTANTE:
                    - Immagina di parlare con una persona comune che soffre di questa malattia.
                    - Usa un linguaggio semplice, evita termini medici troppo tecnici.
                    - Concentrati su sintomi, vita quotidiana, gestione pratica, trattamenti noti, quando rivolgersi al medico.
                    - Non includere informazioni non presenti nel testo.
                    - Ogni domanda deve essere chiara, utile e rilevante per la persona che vive con questa patologia.
                    - Le domande NON devono essere troppo semplici o ovvie.

                    Per ogni domanda:
                    1. Scegli un aspetto concreto e comprensibile (es. cosa fare, cosa evitare, cosa osservare, come si sente il corpo).
                    2. Formula una domanda semplice ma utile su quell'aspetto.
                    3. Fornisci 4 risposte con questi punteggi:
                    - una corretta (5 punti)
                    - molto simile a quella corretta ma con qualcosa di sbagliato (2 punti)
                    - una sbagliata (0 punti)
                    - una molto sbagliata (-2 punti)
                    
                    ## ESEMPIO DI DOMANDA:
                    Domanda: "Quale situazione è tipica dell'incontinenza urinaria da sforzo?"
                    Risposte:
                    - "Perdita di urina durante uno starnuto o quando si sollevano pesi." (5 punti)
                    - "Perdita di urina durante le ore notturne." (2 punti)
                    - "Perdita di urina subito dopo aver bevuto molta acqua." (0 punti)
                    - "Perdita di urina solo mentre si dorme." (-2 punti)
                    ## FINE ESEMPIO

                    Tutte le domande e risposte devono essere in {language}.
                    Non menzionare il testo di riferimento.
                    Assicurati che ogni domanda abbia esattamente 4 risposte.
                    """,
                    output_type=Quiz,
                    model=self.model
                )
                
                try:
                    chunk_quiz_result = await Runner.run(quiz_generator_agent_for_chunk, chunk_summary)
                    if chunk_quiz_result.final_output:
                        chunk_quiz = chunk_quiz_result.final_output_as(Quiz)
                        if chunk_quiz and chunk_quiz.questions:
                            all_questions.extend(chunk_quiz.questions)
                            logging.info(f"Generated {len(chunk_quiz.questions)} questions from chunk {i + 1}. Total questions so far: {len(all_questions)}.")
                    else:
                        logging.warning(f"No output from quiz generator for chunk {i + 1}/{num_chunks} of {filename}.")
                except Exception as e:
                    logging.error(f"Error generating quiz for chunk {i + 1}/{num_chunks} of {filename}: {str(e)}")
                    logging.error(traceback.format_exc())
            
            # save the aggregated summaries
            if aggregated_chunk_summaries_for_saving:
                full_aggregated_summary_text = "\n\n".join(aggregated_chunk_summaries_for_saving)
                summary_filename = f"{base_filename}_summary.txt"
                summary_path = os.path.join(self.summary_dir, summary_filename)
                save_text_to_file(full_aggregated_summary_text, summary_path)
                logging.info(f"Aggregated summary of chunks saved to: {summary_path}")

            if not all_questions:
                logging.error(f"No questions generated for {filename} after processing all chunks.")
                return None, base_filename

            # create final Quiz object from all collected questions
            final_quiz = Quiz(questions=all_questions[:num_questions_total] if num_questions_total > 0 else all_questions)
            logging.info(f"Generated a final quiz with {len(final_quiz.questions)} questions for {filename}.")
            return final_quiz, base_filename
            
        except Exception as e:
            logging.error(f"Error processing {filename} in create_quiz_from_text: {str(e)}")
            logging.error(traceback.format_exc())
            return None, None

# test QuizGenerator
# if __name__ == '__main__':
#     async def test_quiz_generator():
#         logging.basicConfig(level=logging.INFO)
#         # dummy summary directory
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         dummy_summary_dir = os.path.join(current_dir, "test_summaries")
#         os.makedirs(dummy_summary_dir, exist_ok=True)

#         generator = QuizGenerator(model="gpt-4o-mini", summary_dir=dummy_summary_dir)
#         sample_text = "L'influenza è una malattia respiratoria..." * 1000 # long text
#         quiz, filename = await generator.create_quiz_from_text(sample_text, "sample_influenza.txt", "Italiano", 5)
        
#         if quiz:
#             print(f"Quiz generated for {filename}:")
#             for q_idx, question in enumerate(quiz.questions):
#                 print(f"  Q{q_idx+1}: {question.question_text}")
#                 for ans in question.answers:
#                     print(f"    - {ans.text} ({ans.score})")
#         else:
#             print(f"Failed to generate quiz for {filename}.")
        
#         # clean up directory
#         # import shutil
#         # shutil.rmtree(dummy_summary_dir)

#     # asyncio.run(test_quiz_generator())