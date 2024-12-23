//+------------------------------------------------------------------+
//|                                                 testeapi2.py.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include <JAson.mqh>

#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

int flag = -1;
double predicted_value = -1;
string last_prediction_time = "1999-09-08 14:30:00";

MqlDateTime    now;

int OnInit()
  {
//---
   TimeToStruct(TimeTradeServer(), now);
   
   // Obter o tempo atual no formato ISO 8601 usando 'now'
   string startTime = StringFormat("%04d-%02d-%02dT%02d:%02d:%02d", 
                                   now.year, now.mon, now.day, 
                                   now.hour, now.min, now.sec);
                                   
   string mode = "Future";                                
   
   postAPI(_Symbol, 30, 50, startTime, 0, mode);
   
   string response = GetRequest();  // Envia a requisição GET
   ProcessResponse(response);  
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Função POST API                          |
//+------------------------------------------------------------------+
string PostRequest(string url, string body) {
   int timeout = 10;  // Timeout de 10 segundos
   string headers = "Content-Type: application/json\r\n";  // Cabeçalho HTTP
   char data[];  // Corpo da requisição
   char result[];  // Resposta da API
   string result_headers;  // Cabeçalhos da resposta

   StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayRemove(data, ArraySize(data) - 1);  // Remover o caractere nulo extra no final

   // Envia a requisição WebRequest
   int res = WebRequest("POST", url, headers, timeout, data, result, result_headers);
   
   if (res == -1) {
      Print("Erro no WebRequest. Código de erro: ", GetLastError());
      return "";
   }

   // A resposta é retornada como char array, então convertendo para string
   string response = CharArrayToString(result);

   // Exibir a resposta no log
   Print("Resposta da API: ", response);

   return response;
}
void postAPI(string symbol, int predict, int train, string startTime, int lagged_regressor, string mode) {
   string url = "http://127.0.0.1:8000/forecast/";  // URL da sua API
   string body = StringFormat("{\"symbol\":\"%s\",\"predict_size\":%d,\"train_size\":%d,\"start_time\":\"%s\",\"lagged_regressor\":%d,\"mode\":\"%s\"}", symbol, predict, train, startTime, lagged_regressor, mode);

   string response = PostRequest(url, body);  // Envia a requisição
   if (response != "") {
      Print("Resposta da API: ", response);  // Imprime a resposta da API
   } else {
      Print("Falha ao receber a resposta da API.");
   }
   
}





// Função para enviar uma requisição GET para a API
string GetRequest() {
   int timeout = 10;  // Timeout de 10 segundos
   string url ="http://127.0.0.1:8000/forecast/";
   string headers = "Content-Type: application/json\r\n";  // Cabeçalho HTTP
   char data[];  // Corpo da requisição (não utilizado em GET)
   char result[];  // Resposta da API
   string result_headers;  // Cabeçalhos da resposta

   // Envia a requisição WebRequest
   int res = WebRequest("GET", url, headers, timeout, data, result, result_headers);
   
   if (res == -1) {
      Print("Erro no WebRequest. Código de erro: ", GetLastError());
      return "";
   }

   // A resposta é retornada como char array, então convertendo para string
   return CharArrayToString(result);
}

// Função para processar a resposta JSON
void ProcessResponse(string response) {
   CJAVal jsn(NULL, jtUNDEF);  // Criar objeto JSON

   // Desserializa o JSON
   if (jsn.Deserialize(response)) {
      // Atribui os valores das variáveis
      flag = jsn["flag"].ToInt();
      predicted_value = jsn["predicted_value"].ToDbl();
      last_prediction_time = jsn["last_prediction_time"].ToStr();

      // Exibir os dados no log
      Print("Flag: ", flag);
      Print("Predicted Value: ", predicted_value);
      Print("Last Prediction Time: ", last_prediction_time);
      
      
   } else {
      Print("Erro ao analisar o JSON.");
   }
}