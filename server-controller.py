import argparse
import asyncio
import logging
import os
import sys
from threading import Thread
from time import sleep
import warnings

import aiohttp_apispec
from aiohttp_apispec import validation_middleware
from aiohttp import web

import app.api.v2
from app import version
from app.api.rest_api import RestApi
from app.api.v2.responses import apispec_request_validation_middleware
from app.objects.c_ability import Ability
from app.objects.c_adversary import Adversary
from app.objects.c_agent import Agent
from app.objects.c_operation import Operation
from app.objects.c_source import Source
from app.objects.secondclass.c_executor import Executor
from app.objects.secondclass.c_fact import Fact
from app.objects.secondclass.c_link import Link
from app.service.app_svc import AppService
from app.service.auth_svc import AuthService
from app.service.contact_svc import ContactService
from app.service.data_svc import DataService, DATA_BACKUP_DIR
from app.service.knowledge_svc import KnowledgeService
from app.service.event_svc import EventService
from app.service.file_svc import FileSvc
from app.service.learning_svc import LearningService
from app.service.planning_svc import PlanningService
from app.service.rest_svc import RestService
from app.utility.base_object import AppConfigGlobalVariableIdentifier
from app.utility.base_world import BaseWorld
from app.utility.config_generator import ensure_local_config
from emu.db import db_connect, db_cnn
from emu.utils import todict, try_to


def setup_logger(level=logging.DEBUG):
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)-5s (%(filename)s:%(lineno)s %(funcName)s) %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    for logger_name in logging.root.manager.loggerDict.keys():
        if logger_name in ('aiohttp.server', 'asyncio'):
            continue
        else:
            logging.getLogger(logger_name).setLevel(100)
    logging.getLogger("markdown_it").setLevel(logging.WARNING)
    logging.captureWarnings(True)


async def start_server():
    await auth_svc.apply(app_svc.application, BaseWorld.get_config('users'))
    runner = web.AppRunner(app_svc.application)
    await runner.setup()
    await web.TCPSite(runner, BaseWorld.get_config('host'), BaseWorld.get_config('port')).start()


def run_tasks(services):
    loop = asyncio.get_event_loop()
    loop.create_task(app_svc.validate_requirements())
    loop.run_until_complete(data_svc.restore_state())
    loop.run_until_complete(knowledge_svc.restore_state())
    loop.run_until_complete(RestApi(services).enable())
    loop.run_until_complete(app_svc.register_contacts())
    loop.run_until_complete(app_svc.load_plugins(args.plugins))
    print('done loading plugins')
    loop.run_until_complete(data_svc.load_data(loop.run_until_complete(data_svc.locate('plugins', dict(enabled=True)))))
    loop.run_until_complete(app_svc.load_plugin_expansions(loop.run_until_complete(data_svc.locate('plugins', dict(enabled=True)))))
    loop.run_until_complete(auth_svc.set_login_handlers(services))
    loop.create_task(app_svc.start_sniffer_untrusted_agents())
    loop.create_task(app_svc.resume_operations())
    loop.create_task(app_svc.run_scheduler())
    loop.create_task(learning_svc.build_model())
    loop.create_task(app_svc.watch_ability_files())
    loop.run_until_complete(start_server())
    try:
        logging.info('All systems ready.')
        # loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(services.get('app_svc').teardown(main_config_file=args.environment))
    return loop


def init_swagger_documentation(app):
    """Makes swagger documentation available at /api/docs for any endpoints
    marked for aiohttp_apispec documentation.
    """
    warnings.filterwarnings(
        "ignore",
        message="Multiple schemas resolved to the name"
    )
    aiohttp_apispec.setup_aiohttp_apispec(
        app=app,
        title='CALDERA',
        version=version.get_version(),
        swagger_path='/api/docs',
        url='/api/docs/swagger.json',
        static_path='/static/swagger'
    )
    app.middlewares.append(apispec_request_validation_middleware)
    app.middlewares.append(validation_middleware)


if __name__ == '__main__':
    def list_str(values):
        return values.split(',')
    sys.path.append('')
    parser = argparse.ArgumentParser('Welcome to the system')
    parser.add_argument('-E', '--environment', required=False, default='local', help='Select an env. file to use')
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level", default='INFO')
    parser.add_argument('--fresh', action='store_true', required=False, default=False,
                        help='remove object_store on start')
    plugin_options = os.listdir('plugins')
    plugin_options = [p for p in plugin_options if p not in ('manx',)]
    parser.add_argument('-P', '--plugins', required=False, default=plugin_options,
                        help='Start up with a single plugin', type=list_str)
    parser.add_argument('--insecure', action='store_true', required=False, default=True,
                        help='Start caldera with insecure default config values. Equivalent to "-E default".')

    parser.add_argument('-R', '--redishost', required=False, default='', help='')
    parser.add_argument('-S', '--redisport', required=False, default='', help='')
    parser.add_argument('-T', '--redispassword', required=False, default='', help='')
    parser.add_argument('-U', '--session', required=False, default='', help='')
    parser.add_argument('-V', '--host', required=False, default='', help='')

    args = parser.parse_args()
    setup_logger(getattr(logging, args.logLevel))

    if args.insecure:
        logging.warning('--insecure flag set. Caldera will use the default.yml config file.')
        args.environment = 'default'
    elif args.environment == 'local':
        ensure_local_config()

    main_config_path = 'conf/%s.yml' % args.environment
    BaseWorld.apply_config('main', BaseWorld.strip_yml(main_config_path)[0])
    logging.info('Using main config from %s' % main_config_path)
    BaseWorld.apply_config('agents', BaseWorld.strip_yml('conf/agents.yml')[0])
    BaseWorld.apply_config('payloads', BaseWorld.strip_yml('conf/payloads.yml')[0])

    data_svc = DataService()
    knowledge_svc = KnowledgeService()
    contact_svc = ContactService()
    planning_svc = PlanningService(
        global_variable_owners=[
            Executor,
            Agent,
            Link,
            AppConfigGlobalVariableIdentifier
        ]
    )
    rest_svc = RestService()
    auth_svc = AuthService()
    file_svc = FileSvc()
    learning_svc = LearningService()
    event_svc = EventService()

    app_svc = AppService(application=web.Application(client_max_size=5120**2))
    app_svc.register_subapp('/api/v2', app.api.v2.make_app(app_svc.get_services()))
    init_swagger_documentation(app_svc.application)

    if args.fresh:
        logging.info("Fresh startup: resetting server data. See %s directory for data backups.", DATA_BACKUP_DIR)
        asyncio.get_event_loop().run_until_complete(data_svc.destroy())
        asyncio.get_event_loop().run_until_complete(knowledge_svc.destroy())

    
    # endpoints:
    # /emu/red/caldera/queue/
    # /emu/red/caldera/results/time/
    # /emu/red/caldera/log
    # op: time, info => return facts, relationships, and links
    # op: time, apply(link_ids) => execute links

    # only available within current process
    # so safe to be used here.
    if len(args.redishost) > 0:
        os.environ['redis_server'] = args.redishost
    if len(args.redisport) > 0:
        os.environ['redis_port'] = args.redisport
    if len(args.redispassword) > 0:
        os.environ['redis_password'] = args.redispassword

    loop = run_tasks(services=app_svc.get_services())
    db_connect()

    adversary = Adversary(
        name='EMU Red',
        adversary_id='123',
        description='EMU Test Red C2 Controller',
        atomic_ordering=[b.ability_id for b in data_svc.ram['abilities']],
    )
    sc = Source(id='3124', name='test', facts=[Fact(trait='domain.user.name', value='bob')])
    agent = Agent(sleep_min=30, sleep_max=60, watchdog=0, platform='windows', host='WORKSTATION',
                 username='testagent', architecture='amd64', group='red', location=r'C:\Users\Public\test.exe',
                 pid=1234, ppid=123, executors=['psh'], privilege='User', exe_name='test.exe', contact='unknown',
                 paw='testpaw')
    op = Operation(
        name='EMU', 
        agents=[agent], 
        adversary=adversary,
        source=sc,
    )
    session = args.session
    host = args.host
    
    new_loop = asyncio.new_event_loop()
    def serve_queue():
        print('serving queue...')
        while True:
            with try_to() as errors:
                command = db_cnn.queue_pop(f'/emu/{session}/{host}/caldera/queue')
                if command:
                    timestamp, action, params = command
                    if action == 'info':
                        links =  new_loop.run_until_complete(planning_svc.get_links(
                            operation=op,
                        ))
                        results =  todict({
                            **knowledge_svc.base_service.fact_ram,
                            'links': links,
                            'agents': [a.schema.dump(a) for a in op.agents],
                        })
                    elif action == 'apply':
                        link_ids = [new_loop.run_until_complete(
                            op.apply(l)) for l in params['links']]
                        new_loop.run_until_complete(
                            op.wait_for_links_completion(link_ids))
                        results = {
                            'done'
                        }
                    results['command'] = command
                    timestamp = timestamp.strftime('%Y%m%d-%H:%M:%S-%f')
                    db_cnn.set(f'/emu/{session}/{host}/caldera/results/{timestamp}/', results)
                sleep(1)
            if len(errors)>0:
                print(str(errors[0]))
                db_connect()

    t = Thread(target=serve_queue)
    t.daemon = True
    t.start()    
    # links =  asyncio.get_event_loop().run_until_complete(planning_svc.get_links(
    #     operation=op, 
    # ))
    # print(links)
    loop.run_forever()








