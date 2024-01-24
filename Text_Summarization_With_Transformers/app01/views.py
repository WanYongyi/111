from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# from Transformer_Model import summarize


from Transformer_Model import preprocess, transformer, train
from Transformer_Model.util import create_masks
import tensorflow as tf
import numpy as np


# Create your views here.


def creat_transformer(num_layers, d_model, num_heads, dff, encoder_vocab_size, decoder_vocab_size):
    transformer_m = transformer.Transformer(
        num_layers,
        d_model,
        num_heads,
        dff,
        encoder_vocab_size,
        decoder_vocab_size,
        pe_input=encoder_vocab_size,
        pe_target=decoder_vocab_size,
    )
    return transformer_m


def Load_Checkpoint(transformer_m):
    # 实例化一个CheckpointManager
    checkpoint_path = "D:\Pycharm\Projects\Text_Summarization_With_Transformers\Transformer_Model\checkpoints_train"
    ckpt = tf.train.Checkpoint(transformer=transformer_m,
                               optimizer=train.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 恢复最新的checkpoint
    if ckpt_manager.latest_checkpoint:
        # ckpt.restore(ckpt_manager.latest_checkpoint)
        # 这些警告表明在尝试从Checkpoint中恢复模型时，发现了一些变量无法找到对应的恢复对象。这通常发生在以下情况下
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        status.expect_partial()
        print("Loaded latest checkpoint")


def evaluate(input_document):
    encoder_maxlen = 400
    decoder_maxlen = 75
    num_layers = 4  # 编码器和解码器堆叠的层数
    d_model = 128  # 嵌入层和编码器/解码器层的维度大小
    dff = 512  # 前馈神经网络中间层的维度大小
    num_heads = 8  # 多头自注意力机制中注意力头的数量
    encoder_vocab_size = preprocess.encoder_vocab_size
    decoder_vocab_size = preprocess.decoder_vocab_size

    transformer_m = creat_transformer(num_layers, d_model, num_heads, dff, encoder_vocab_size, decoder_vocab_size)
    Load_Checkpoint(transformer_m)

    input_document = preprocess.document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen,
                                                                   padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [preprocess.summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        predictions, attention_weights = transformer_m(
            encoder_input,
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == preprocess.summary_tokenizer.word_index["<stop>"]:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    result = preprocess.summary_tokenizer.sequences_to_texts(summarized)[0]
    print(result)
    return result  # since there is just one translated document


@csrf_exempt
def index(request):
    if request.method == "GET":
        return render(request, 'index.html')

    text = request.POST.get('text')
    result = summarize(text)
    # result = len(text)
    return JsonResponse({"status": True, "summarize": result})
