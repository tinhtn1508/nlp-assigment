# nlp-assigment
This is a assigment for nlp subject
# train
There two approaches
  - For word embedding (generate sentences):\
    python main.py --cuda --embsize 1000 --nhidden 1000 --dropout 0.65 --epochs 40 --sequence_length 20
    
  - For character embedding (check grammar):\
    python main.py --cuda --embsize 100 --nhidden 100 --dropout 0.3 --epochs 40 --batch_size 500
 # check-grammar and auto correct
  Run this command to see output: python check_grammar.py --cuda --prefix 'hậu vu'\
  Result:\
        h --> ậ -- probablity: 2.64%\
        ậ --> u -- probablity: 1.74%\
        u -->   -- probablity: 95.77%\
          --> v -- probablity: 98.26%\
        v --> u -- probablity: 0.01% -- candidate: ớ, ệ, à, ề, i\
 # generate sentences
 Execute: python generate.py --cuda --prefix "cầu thủ eden hazard" --nsentence 10\
 Result:\
      cầu thủ eden hazard đá chính .\
      với việc là một trong những nơi ở old traffordman utd sẽ vượt lên sau hai thất bại gần đây họ rồi .\
      thất bại của man utd như ở trận đầu tiên của họ .\
      cơ hội cuối cùng là trận đấu giữa real và barca .\
      ronaldo chia sẻ quan điểm của cậu ấy .\
      trước đây tôi nghĩ anh ấy không có nhiều cầu thủ trở thành một kẻ nhỏ .\
      đó là lý do khiến messi ra nhiều .\
      real muốn mua malcom trong hè 2020 .\
      một trong những cầu thủ lớn nhất châu âu như kylian mbappe và javier mascherano là cristiano ronaldo .\
      tính từ mùa trướccầu thủ ghi bàn nhiều nhất mùa giải 2017-2018 .\
