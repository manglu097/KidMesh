import logging
import wandb
import time

logger = logging.getLogger(__name__)


class Trainer(object):

    def training_step(self, data, epoch):
        self.optimizer.zero_grad()
        loss, log = self.net.module.loss(data, epoch)
        loss.backward()
        self.optimizer.step()

        return log

    def __init__(self, net, trainloader, optimizer, numb_of_itrs, eval_every, save_path, evaluator):
        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.numb_of_itrs = numb_of_itrs
        self.eval_every = eval_every
        self.save_path = save_path
        self.evaluator = evaluator

    def train(self, start_iteration=1):
        print("Start training...")
        start_time = time.time()
        self.net = self.net.train()
        iteration = start_iteration
        for epoch in range(10000000):  # loop over the dataset multiple times

            start_time3 = time.time()
            for itr, data in enumerate(self.trainloader):
                loss = self.training_step(data, iteration)  # start_iteration 没啥用
                wandb.log(loss)
                iteration = iteration + 1
                if iteration % self.eval_every == self.eval_every - 1:  # print every K epochs
                    start_time2 = time.time()
                    self.evaluator.evaluate(iteration)
                    end_time2 = time.time()
                    training_time = end_time2 - start_time2
                    hours = training_time // 3600
                    minutes = (training_time % 3600) // 60
                    seconds = training_time % 60

                    # 集成到print语句里，输出要有单位
                    print(f"单次评价时间为：{hours}小时{minutes}分钟{seconds}秒")


                if iteration > self.numb_of_itrs:
                    break  # 尽管跳出最内层循环，但实际是结束程序


            end_time3 = time.time()
            training_time = end_time3 - start_time3
            hours = training_time // 3600
            minutes = (training_time % 3600) // 60
            seconds = training_time % 60

            # 集成到print语句里，输出要有单位
            print(f"单个训练集训练时间为：{hours}小时{minutes}分钟{seconds}秒")

            if iteration > self.numb_of_itrs:
                break  # 尽管跳出最内层循环，但实际是结束程序
        end_time = time.time()
        training_time = end_time - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        seconds = training_time % 60

        # 集成到print语句里，输出要有单位
        print(f"总训练时间为：{hours}小时{minutes}分钟{seconds}秒")
        
        logger.info("... end training!")
