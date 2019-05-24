import os
import json
import logging
import tensorflow as tf
import numpy as np
from telegram import ChatAction

import sample
import model
import encoder
import telegram
import random
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, BaseFilter

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def gpt2_session_and_out_op(seed=None, model_name='345M', length=75, temperature=1, top_k=40):
    batch_size = 1
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        context = tf.placeholder(tf.int32, [batch_size, None], name='context')
        np.random.seed(seed)
        tf.set_random_seed(seed)
        out_op = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

    return sess, out_op


def gpt2_generate(raw_text, sess, out_op, model_name='345M'):
    this_dir = os.path.dirname(os.path.realpath(__file__))

    enc = encoder.get_encoder(model_name, os.path.join(this_dir, "../models/"))
    context_tokens = enc.encode(raw_text)
    out = sess.run(out_op, feed_dict={
        'context:0': [context_tokens]})
    out_tokens = out[:, len(context_tokens):]
    text = enc.decode(out_tokens[0])
    return text


def start(update: telegram.Update, context: telegram.ext.CallbackContext):
    update.message.reply_text("Hello I'm a bot that conditions the GPT2-345M"
                              " on your messages. I will reply to you by "
                              "continuing the message you sent me.")


def reply_with_gpt2(update: telegram.Update, context: telegram.ext.CallbackContext, gpt2_output):
    logger.info("Received a message.")
    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    text = update.message.text.replace("@{}".format(context.bot.username), "")
    output = gpt2_output(text)

    punctuation_characters = "!?."
    last_occurrences = [output.rfind(x) for x in punctuation_characters]
    if not all([x == -1 for x in last_occurrences]):
        output = output[:max(last_occurrences) + 1]
    output = output.split("<|endoftext|>")[0]
    update.message.reply_text(output, quote=True)


class HandleMentionsOnly(BaseFilter):
    def filter(self, message: telegram.Message):
        return message.bot.username in message.text


class HandleRandomly(BaseFilter):
    def filter(self, update):
        return random.random() <= 0.10


def main():
    token = os.environ.get("TELEGRAM_BOT_SECRET")
    if token is None:
        logger.error("You need to set the environment variable TELEGRAM_BOT_SECRET to be able to run this script.")
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))

    sess, out_op = gpt2_session_and_out_op()

    def reply(update: telegram.Update, context: telegram.ext.CallbackContext):
        return reply_with_gpt2(update, context, lambda x: gpt2_generate(x, sess, out_op))

    dp.add_handler(MessageHandler(
        (Filters.text &
         (Filters.private | (Filters.group & HandleRandomly()) | HandleMentionsOnly())),
        reply))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
