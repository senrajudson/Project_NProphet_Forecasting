//+------------------------------------------------------------------+
//|                                                  trendMA-RSI.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh> //importando Classe Trade (CTrade)
#include <stdliberr.mqh> // Para gerenciar erros de WebRequest
#include <JAson.mqh>

// Import CTrade
CTrade trade;
// URLs da API
//#define BASE_URL      "http://127.0.0.1:8000"
//#define FORECAST_POST "/forecast"
//#define FORECAST_GET  "/forecast"
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input ulong          INP_VOLUME        = 1;              // volume
input int            INP_TRAIN_SIZE    = 50;           // train size
input int            INP_PREDICT_SIZE  = 30;           // predict size
input string         INP_MODE          = "Future";        // mode || Future || Lag
input int            INP_LAG           = 0;            // lag_regressor para mode lag
input double         INP_TP            = 100;            // take profit
input double         INP_SL            = 8;             // stop loss
input double         INP_DOWNGPROFIT   = 0.75;            // downgrading max profit
input double         INP_INITDOWNG     = 40;             // start downgrading profit count
input double         INP_POSPROFIT     = 5000;            // max profit position
input double         INP_POSLOSS       = -500;           // max loss position
input ulong          INP_MAGIC         = 323232;         // magic number
input double         INP_DAYPROFIT     = 5000;            // max profit today
input double         INP_DAYLOSS       = -1000;           // max loss today

//+------------------------------------------------------------------+
//| Global var                                                       |
//+------------------------------------------------------------------+

int flag = -1;
double predicted_value = -1;
string last_prediction_time = "1999-09-08 14:30:00";
string used_prediction_time = "1999-09-08 00:00:00";

double         maxProfit[];
int            lastProcessedDay = -1; // é mais por causa do tester, verificar se o dia mudou

MqlTick        tick;
MqlRates       rates[];
MqlDateTime    now;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
      
      trade.SetExpertMagicNumber(INP_MAGIC);
      
      ArraySetAsSeries(rates, true);
      
      // Configura o buffer como um array dinâmico
      ArrayResize(maxProfit, 1);
      ArraySetAsSeries(maxProfit, true); // Torna o array como uma série (ordem inversa)
      maxProfit[0] = 0.0;
      
      datetime current_time = TimeTradeServer();
      TimeToStruct(TimeTradeServer(), now);
      
      // Obter o tempo atual no formato ISO 8601 usando 'now'
      string startTime = StringFormat("%04d-%02d-%02dT%02d:%02d:%02d", 
                                      now.year, now.mon, now.day, 
                                      now.hour, now.min, now.sec);
                                      
      string mode = "Future";                                
      
      postAPI(_Symbol, 30, 50, startTime, 0, mode);
      
      string response = GetRequest();  // Envia a requisição GET
      ProcessResponse(response);  
      
       // Converter a string para o tipo datetime
      
      Print("Last predicted time: ", last_prediction_time, 
            " Current time: ", current_time);      

      EventSetTimer(1);

//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
    EventKillTimer();

    ObjectsDeleteAll(ChartID(), "CandleTime");

    ChartRedraw();

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---      
   
      datetime current_time = TimeTradeServer();
      TimeToStruct(current_time, now);
      PositionSelect(_Symbol); // número de posições abertas
      
      // Obter o tempo atual no formato ISO 8601 usando 'now'
      string startTime = StringFormat("%04d-%02d-%02dT%02d:%02d:%02d", 
                                      now.year, now.mon, now.day, 
                                      now.hour, now.min, now.sec);      

      if ((now.hour*60)+now.min > (17*60)+50) CloseOrders(INP_MAGIC); //isso não funciona no tester | agora vai funcionar!
      
      if ((now.hour*60)+now.min > (17*60)+30) return;
      
      //if (now.day_of_week == WEDNESDAY) return;    
      
      int copied_rates = CopyRates(_Symbol, _Period, 0, 5, rates);
      if (copied_rates < 4) return;      
      
      if (!SymbolInfoTick(_Symbol, tick)) return;  
      
      //check openned positions profit
      double positionProfit = GetProfitByPosition(INP_MAGIC);
      
      if(positionProfit > maxProfit[0]) maxProfit[0] = positionProfit;
      
      //Print(maxProfit[0]);
      double posLoss = INP_POSLOSS;
      double posProfit = INP_POSPROFIT;
      
      Print(maxProfit[0], " ", positionProfit);
      
      //if(maxProfit[0] > INP_INITDOWNG && (positionProfit/maxProfit[0]) < INP_DOWNGPROFIT) CloseOrders(INP_MAGIC); 
      //if(positionProfit > posProfit) CloseOrders(INP_MAGIC);
      //if(positionProfit < posLoss) CloseOrders(INP_MAGIC);
      
      if(maxProfit[0] > INP_INITDOWNG && (positionProfit/maxProfit[0]) < INP_DOWNGPROFIT) // Profit management
       {
            CloseOrders(INP_MAGIC);
            postAPI(_Symbol, INP_PREDICT_SIZE, INP_TRAIN_SIZE, startTime, INP_LAG, INP_MODE);
            maxProfit[0] = 0.0;
        }
        
      if(positionProfit > posProfit) // profit    
        {
            CloseOrders(INP_MAGIC);
            postAPI(_Symbol, INP_PREDICT_SIZE, INP_TRAIN_SIZE, startTime, INP_LAG, INP_MODE);
            maxProfit[0] = 0.0;
        }
      if(positionProfit < posLoss) // loss
        {
            CloseOrders(INP_MAGIC);
            postAPI(_Symbol, INP_PREDICT_SIZE, INP_TRAIN_SIZE, startTime, INP_LAG, INP_MODE);
            maxProfit[0] = 0.0;
        }      
      
      double todayProfits = GetProfitByDeals(INP_MAGIC);
      if(todayProfits > INP_DAYPROFIT) return;
      if(todayProfits < INP_DAYLOSS) return;       
      
      if (!CheckForNewCandle()) return; // se não houver novo candle
      
      string response = GetRequest();  // Envia a requisição GET
      ProcessResponse(response);       
                                      
      // Converter a string para o tipo datetime
      
      Print("Last predicted time: ", last_prediction_time, 
            " Current time: ", current_time);
      
      if (current_time > last_prediction_time) {
         postAPI(_Symbol, INP_PREDICT_SIZE, INP_TRAIN_SIZE, startTime, INP_LAG, INP_MODE);
         return;
      }
      
      //if (predicted_value - rates[1].close > -8 && predicted_value - rates[1].close < 8) {
      //   postAPI(_Symbol, INP_PREDICT_SIZE, INP_TRAIN_SIZE, startTime, INP_LAG, INP_MODE);
      //   return;
      //}
            
      // Verificar se o dia mudou
      if (now.day != lastProcessedDay)
      {
        // Resetar o maxProfit 
        maxProfit[0] = 0.0;
        lastProcessedDay = now.day;
      }      
      
      if (PositionsTotal() > 0) return; // verifica se há alguma posição aberta || modificar para ser por magic number
      if (used_prediction_time == last_prediction_time) return;
      
      // buy
      if (predicted_value - rates[1].close > 5 && flag == 1)
        {
            
            trade.Buy(INP_VOLUME, _Symbol, tick.last, tick.last - INP_SL, 0, "buy forecast");
            Print("buy");
            if (used_prediction_time != last_prediction_time) used_prediction_time = last_prediction_time;
            return;
        }
      
      // sell
      if (rates[1].close - predicted_value > 5 && flag == 0)
        {
        
            trade.Sell(INP_VOLUME, _Symbol, tick.last, tick.last + INP_SL, 0, "sell forecast");
            Print("sell");
            if (used_prediction_time != last_prediction_time) used_prediction_time = last_prediction_time;
            return;
        }       
      
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---
      int copied_rates = CopyRates(_Symbol, _Period, 0, 5, rates);
      
      if (copied_rates < 4) return;
    

      ShowTime(rates[0]);
   
  }
//+------------------------------------------------------------------+
//| Custom function                                                  |
//+------------------------------------------------------------------+

void CloseOrders(ulong _magicNumber) {
   
   ulong orderTicket = 0;
   int try = 100; // uma segurança para não travar a função no while
   // o uso do while e não 'for' é porque a requisição de delete pode retornar como inválida, portanto não seria deletado
   while(OrdersTotal() != 0 && try-- > 0) // enquanto houver ordens pendentes, ficar dentro do while
     { //no while, se retornar requisição inválida ele continua tentando deletar até conseguir
         orderTicket       = OrderGetTicket(0);
         ulong    magic    = OrderGetInteger(ORDER_MAGIC);
         if(magic == _magicNumber)
         {
               trade.OrderDelete(orderTicket); // deletar ordens
         }
     }
     
   try = 100; // reiniciar tentativas
     
   while(PositionsTotal() != 0 && try-- > 0) // enquanto houver ordens pendentes, ficar dentro do while
     { //no while, se retornar requisição inválida ele continua tentando deletar até conseguir
         ulong    positionTicket    = PositionGetTicket(0); // Obter o ticket da primeira posição
         ulong    magic             = PositionGetInteger(POSITION_MAGIC);
         double   volume            = PositionGetDouble(POSITION_VOLUME);
         
         if(magic == _magicNumber && volume > 0)
           {
               trade.PositionClose(positionTicket);
           }
         
     }     
   
}
//+------------------------------------------------------------------+
//| Check for new candle                                             |
//+------------------------------------------------------------------+
bool CheckForNewCandle() {

   static datetime previosTime = 0;
   
   datetime currentTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(previosTime!=currentTime)
     {
         previosTime=currentTime;
         
         return true;
     }
   
   return false;
}
//+------------------------------------------------------------------+
//| Display candle time                                              |
//+------------------------------------------------------------------+
void ShowTime(MqlRates & _rates) {

   string name = "CandleTime";
   
   string time = StringSubstr(TimeToString(PeriodSeconds() - (TimeCurrent() - _rates.time), TIME_SECONDS), 3);
   
   ObjectCreate(ChartID(), name, OBJ_TEXT, 0, TimeCurrent() + (PeriodSeconds()), _rates.open  );
   
   ObjectSetString(ChartID(), name, OBJPROP_TEXT, time);
   
   ObjectSetInteger(ChartID(), name, OBJPROP_COLOR, clrYellow);
   
}
//+------------------------------------------------------------------+
//| Get profit in history                                            |
//+------------------------------------------------------------------+
double GetProfitByDeals(ulong _magicNumber) {

      datetime start   = StringToTime(TimeToString(TimeCurrent(), TIME_DATE)); // Início do dia atual
      datetime end     = TimeCurrent(); // Hora atual
      double   profits = 0.0;
      
      if (HistorySelect(start, end)) {

            for (int i = 0; i < HistoryDealsTotal(); i++) {
                  
                  ulong dealTicket = HistoryDealGetTicket(i); // Obter o ticket do negócio
                  
                  ulong dealMagic = HistoryDealGetInteger(dealTicket, DEAL_MAGIC);
                  
                  if(dealMagic == _magicNumber)
                   {
                        double   dealProfit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
                        profits += dealProfit;
                   }
                         
            }
      }   

      return profits;

}
//+------------------------------------------------------------------+
//| Get profit in positions openned                                  |
//+------------------------------------------------------------------+
double GetProfitByPosition(ulong _magicNumber) {

      double   profits = 0.0;

      for (int i = 0; i < PositionsTotal(); i++) {
            
            //ulong posTicket = PositionGetTicket(i); // Obter o ticket do negócio
            ulong posMagic  = PositionGetInteger(POSITION_MAGIC);
            
            if (posMagic == _magicNumber) {
                  double   posProfit = PositionGetDouble(POSITION_PROFIT);
                  profits += posProfit;
                       
            }
      }  

      return profits;

}

//+------------------------------------------------------------------+
//| Função POST API                          |
//+------------------------------------------------------------------+
string PostRequest(string url, string body) {
   int timeout = 400;  // Timeout de 10 segundos
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
//+------------------------------------------------------------------+
//| Função para fazer uma requisição GET                             |
//+------------------------------------------------------------------+
string GetRequest() {
   int timeout = 400;  // Timeout de 10 segundos
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
      string predicted_time = jsn["last_prediction_time"].ToStr();
      datetime predicted_time_datetime = predicted_time;
      last_prediction_time = predicted_time;

      // Exibir os dados no log
      Print("Flag: ", flag);
      Print("Predicted Value: ", predicted_value);
      Print("Last Prediction Time: ", last_prediction_time);
      
      
   } else {
      Print("Erro ao analisar o JSON.");
   }
}
//+------------------------------------------------------------------+