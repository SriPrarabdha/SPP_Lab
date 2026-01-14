from tasks.task_1 import run_task_1
from tasks.task_2 import run_task_2
from tasks.task_3 import run_task_3
from argparse import ArgumentParser


def main(tasks:list[int]):

    tasks_list = {
        1: ["ip_images/coins.png" , run_task_1],
        2: ["ip_files/page.png" , run_task_2],
        3: ["ip_files/flowers.png" , run_task_3],
    }

    for key in tasks:
        tasks_list[key][1](tasks_list[key][0])


if __name__ == "__main__":
    parser = ArgumentParser("Parser")
    parser.add_argument(
        "--task",
        type=int,
        nargs="+",            
        default=[1,2,3],         
        help="Which task(s) do you want to run? Default is all."
    )
    args = parser.parse_args()
    
    main(args.task) 

    


