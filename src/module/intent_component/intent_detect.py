from utils.logger import logger
from module.openai_service.completion import answer_decider, call_completion
from fastapi import HTTPException
from fastapi.responses import JSONResponse


class IntentComponent:

    def detect_from_sender_histories(self, sender_data: str = "",
                                     conversation_history: list = []):
        try:
            response_data = answer_decider(question=sender_data,
                                            histories=conversation_history)
            return JSONResponse(content=response_data, status_code=200)
        except Exception as e:
            logger.error(
                "Exception in detect from sender and histories: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
